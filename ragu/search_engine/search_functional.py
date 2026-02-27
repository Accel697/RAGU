# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/

from dataclasses import asdict
from typing import List

from ragu.common.prompts.default_models import SubQuery
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Entity, Community


async def _find_most_related_edges_from_entities(entities: list[Entity], knowledge_graph: KnowledgeGraph):
    entity_ids = [entity.id for entity in entities if entity and entity.id]
    if not entity_ids:
        return []

    grouped_edges = await knowledge_graph.index.graph_backend.get_all_edges_for_nodes(entity_ids)
    all_related_edges = [edge for edges in grouped_edges for edge in edges if edge]

    if not all_related_edges:
        return []

    all_edges_data = []
    for edge in all_related_edges:
        edge_data = asdict(edge)
        all_edges_data.append(edge_data)

    all_edges_data = sorted(
        all_edges_data,
        key=lambda x: (x["relation_strength"]),
        reverse=True
    )

    return all_edges_data


async def _find_most_related_text_unit_from_entities(
        entities: List[Entity],
        knowledge_graph: KnowledgeGraph
):
    chunks_id = [entity.source_chunk_id for entity in entities]

    edges = await knowledge_graph.index.graph_backend.get_all_edges_for_nodes([e.id for e in entities])

    grouped_relations = await knowledge_graph.index.graph_backend.get_all_edges_for_nodes([e.id for e in entities])
    relations = [relation for group in grouped_relations for relation in group if relation]
    neighbor_ids = list(set([relation.object_id for relation in relations]))
    neighbors = await knowledge_graph.index.get_entities(neighbor_ids)

    all_one_hop_text_units_lookup = {
        neighbor.id : neighbor.source_chunk_id for neighbor in neighbors if neighbor is not None
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(chunks_id, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                        e.object_id in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e.object_id]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await knowledge_graph.index.chunks_kv_storage.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    chunks = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = [t["data"] for t in chunks]
    return all_text_units

async def _find_documents_id(entities: List[Entity]):
    documents_set = set()
    for entity in entities:
        if hasattr(entity, 'documents_id') and entity.documents_id:
            documents_set.update(entity.documents_id)
    return list(documents_set)


async def _find_most_related_community_from_entities(
        entities: List[Entity],
        knowledge_graph: KnowledgeGraph,
        level: int = 2
):
    if not entities:
        return []

    desired_community_ids: set[str] = set()
    for entity in entities:
        if not getattr(entity, "clusters", None):
            continue
        for cluster_data in entity.clusters:
            try:
                c_level = int(cluster_data.get("level", 9999))
            except Exception:
                continue
            if c_level <= level:
                cid = cluster_data.get("cluster_id")
                if cid is None:
                    continue

                cid_str = str(cid)
                if cid_str.startswith("com-"):
                    desired_community_ids.add(cid_str)
                    continue

                try:
                    cluster_id = int(cid_str)
                except Exception:
                    continue

                community_id = Community(
                    level=c_level,
                    cluster_id=cluster_id,
                    entities=[],
                    relations=[]
                ).id
                desired_community_ids.add(community_id)

    if not desired_community_ids:
        return []

    summary_store = knowledge_graph.index.community_summary_kv_storage

    summaries = await summary_store.get_by_ids(list(desired_community_ids))
    final_summaries = [s for s in summaries if s]

    return final_summaries

def _topological_sort(subqueries: List[SubQuery]) -> List[SubQuery]:
    by_id = {q.id: q for q in subqueries}
    visited = set()
    ordered: List[SubQuery] = []

    def visit(q: SubQuery):
        if q.id in visited:
            return
        for dep in q.depends_on:
            visit(by_id[dep])
        visited.add(q.id)
        ordered.append(q)

    for q in subqueries:
        visit(q)

    return ordered
