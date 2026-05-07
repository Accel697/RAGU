import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from ragu import (
    KnowledgeGraph,
    LocalSearchEngine,
    SimpleChunker,
    BuilderArguments,
    Settings,
    ArtifactsExtractorLLM,
)
from ragu.models.llm import LLMOpenAI
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.utils.ragu_utils import read_text_from_files
from ragu.storage.graph_storage_adapters.age_adapter import AgeGraphStorage, AgeNode, AgeEdge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAGU API")

storage: Optional[AgeGraphStorage] = None
knowledge_graph: Optional[KnowledgeGraph] = None
local_search: Optional[LocalSearchEngine] = None


class QueryRequest(BaseModel):
    question: str


class GraphRecord(BaseModel):
    id: str
    label: str
    properties: dict


class EdgeRecord(BaseModel):
    id: str
    subject_id: str
    object_id: str
    label: str
    properties: dict


def get_ollama_client() -> CachedAsyncOpenAI:
    """Create Ollama client compatible with RAGU OpenAI wrapper"""
    base_url = os.getenv("OLLAMA_BASE_URL")
    return CachedAsyncOpenAI(
        base_url=base_url,
        api_key="ollama",
        rate_min_delay=1,
        rate_max_simultaneous=5,
        retry_times_sec=(1, 2, 2),
        cache="./llm_cache",
    )


async def _pull_ollama_model(api_url: str, model_name: str) -> None:
    """Pull model into Ollama container if not present"""
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            await client.post(f"{api_url}/api/pull", json={"name": model_name, "stream": False})
            logger.info(f"Model {model_name} ready")
        except Exception as e:
            logger.warning(f"Model pull skipped/failed (may already exist): {e}")


async def init_ragu_components() -> None:
    """Initialize RAGU pipeline"""
    global storage, knowledge_graph, local_search

    storage = AgeGraphStorage(
        host=os.getenv("AGE_HOST"),
        port=int(os.getenv("AGE_PORT")),
        database=os.getenv("AGE_DB"),
        user=os.getenv("AGE_USER"),
        password=os.getenv("AGE_PASSWORD"),
        graph_name=os.getenv("GRAPH_NAME"),
    )
    await storage.index_start_callback()
    logger.info("Connected to Apache AGE")

    client = get_ollama_client()
    llm_model = os.getenv("LLM_MODEL")
    embed_model = os.getenv("EMBED_MODEL")
    ollama_api = os.getenv("OLLAMA_API").rstrip("/v1")

    await _pull_ollama_model(ollama_api, llm_model)
    await _pull_ollama_model(ollama_api, embed_model)

    llm = LLMOpenAI(client, model_name=llm_model)
    embedder = EmbedderOpenAI(client, model_name=embed_model, dim=768)

    Settings.storage_folder = "/app/ragu_working_dir"
    Settings.language = "russian"

    chunker = SimpleChunker(max_chunk_size=1000)
    artifact_extractor = ArtifactsExtractorLLM(llm=llm, do_validation=False)
    builder_args = BuilderArguments(
        use_llm_summarization=True,
        vectorize_chunks=True,
        make_community_summary=False,
        remove_isolated_nodes=True,
    )

    knowledge_graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=chunker,
        artifact_extractor=artifact_extractor,
        builder_settings=builder_args,
    )

    public_attrs = [a for a in dir(knowledge_graph) if not a.startswith('_')]
    logger.info(f"KnowledgeGraph public attributes: {public_attrs}")

    for attr_name in ['entities', 'relations', 'chunks', '_entities', '_relations', '_chunks', 'graph', '_graph']:
        if hasattr(knowledge_graph, attr_name):
            attr_value = getattr(knowledge_graph, attr_name)
            attr_type = type(attr_value).__name__
            attr_len = len(attr_value) if hasattr(attr_value, '__len__') else 'N/A'
            logger.info(f"Found attribute '{attr_name}': type={attr_type}, len={attr_len}")

    local_search = LocalSearchEngine(
        llm=llm,
        knowledge_graph=knowledge_graph,
        embedder=embedder,
        tokenizer_model="gpt-3.5-turbo",
    )
    logger.info("RAGU components initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_ragu_components()
    yield
    if storage:
        await storage.close()
        logger.info("Disconnected from Apache AGE")


app.router.lifespan_context = lifespan


@app.get("/graph/records")
async def get_records():
    """Return all nodes and edges"""
    if not storage:
        return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}

    nodes = await storage.get_all_nodes()
    node_records = [
        GraphRecord(id=n.id, label=n.label, properties=n.properties)
        for n in nodes
    ]

    edges = await storage.get_all_edges()
    edge_records = [
        EdgeRecord(
            id=e.id,
            subject_id=e.subject_id,
            object_id=e.object_id,
            label=e.label,
            properties=e.properties
        )
        for e in edges
    ]

    return {
        "nodes": node_records,
        "edges": edge_records,
        "total_nodes": len(node_records),
        "total_edges": len(edge_records)
    }


@app.post("/graph/load")
async def load_data():
    """Index text files from data/ru/ into the knowledge graph"""
    if not knowledge_graph:
        return {"status": "error", "message": "Knowledge graph not initialized"}

    xdt_url = "http://xdt-demo:8081/collect"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(xdt_url)
            response.raise_for_status()
            collected = response.json()
    except Exception as e:
        return {"status": "error", "message": f"Failed to fetch from xdt_mgr: {e}"}

    docs = []
    for i, item in enumerate(collected):

        raw = item.get("rawData") or item.get("raw_data") or item.get("RawData")


        if raw and not str(raw).startswith("Error:"):
            docs.append(str(raw))

    if not docs:
        return {"status": "no_data", "message": "No valid data received from sources"}

    await knowledge_graph.build_from_docs(docs)

    entities_saved = 0
    relations_saved = 0

    # === SYNC ENTITIES: RAGU internal -> Apache AGE ===
    internal_entities = await knowledge_graph.index.graph_backend.get_all_nodes()
    if internal_entities:
        age_nodes = []
        for entity in internal_entities:
            node = AgeNode(
                id=str(entity.id),
                label=str(getattr(entity, 'entity_type', 'Entity')),
                properties={
                    'name': getattr(entity, 'entity_name', str(entity.id)),
                    'description': getattr(entity, 'description', ''),
                    'source_chunks': getattr(entity, 'source_chunk_id', []),
                    'documents': getattr(entity, 'documents_id', []),
                }
            )
            age_nodes.append(node)

        if age_nodes:
            await storage.upsert_nodes(age_nodes)
            entities_saved = len(age_nodes)
            logger.info(f"Saved {entities_saved} entities to AGE via upsert_nodes")

    # === SYNC RELATIONS: RAGU internal -> Apache AGE ===
    internal_relations = await knowledge_graph.index.graph_backend.get_all_edges()
    if internal_relations:
        age_edges = []
        for relation in internal_relations:
            edge = AgeEdge(
                id=str(relation.id),
                subject_id=str(relation.subject_id),
                object_id=str(relation.object_id),
                label=str(getattr(relation, 'relation_type', 'RELATED_TO')),
                properties={
                    'description': getattr(relation, 'description', ''),
                    'strength': getattr(relation, 'relation_strength', 1.0),
                    'subject_name': getattr(relation, 'subject_name', ''),
                    'object_name': getattr(relation, 'object_name', ''),
                }
            )
            age_edges.append(edge)

        if age_edges:
            await storage.upsert_edges(age_edges)
            relations_saved = len(age_edges)
            logger.info(f"Saved {relations_saved} relations to AGE via upsert_edges")

    logger.info(f"Sync complete: {entities_saved} entities, {relations_saved} relations saved to AGE")

    return {
        "status": "ok",
        "docs_processed": len(docs) if docs else 0,
        "entities_saved": entities_saved,
        "relations_saved": relations_saved
    }


@app.post("/query")
async def query(req: QueryRequest):
    """Execute RAG query via LocalSearchEngine"""
    if not local_search:
        return {"error": "Search engine not initialized"}

    result = await local_search.a_query(req.question)

    return {
        "question": req.question,
        "answer": result.response if hasattr(result, "response") else str(result),
        "engine": "local_search",
        "language": "russian"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)