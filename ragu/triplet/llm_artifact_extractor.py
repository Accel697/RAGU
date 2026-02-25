from __future__ import annotations

from typing import Any, List, Tuple, Optional, cast

from pydantic import BaseModel

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render
from ragu.graph.types import Entity, Relation
from ragu.llm.llm import LLM
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor
from ragu.triplet.types import NEREL_ENTITY_TYPES, NEREL_RELATION_TYPES


class ArtifactsExtractorLLM(BaseArtifactExtractor):
    """
    Extracts entities and relations from text chunks using LLM.

    Pipeline:
      1. Render the `artifact_extraction` instruction in batch mode over chunk texts.
      2. Call the LLM to produce structured artifacts for each chunk.
      3. Optionally render and run `artifact_validation` to refine extracted artifacts.
      4. Convert model outputs into Entity/Relation objects, preserving source chunk ids.
    """

    def __init__(
        self,
        llm: LLM,
        do_validation: bool = False,
        language: str | None = None,
        entity_types: Optional[List[str]] = NEREL_ENTITY_TYPES,
        relation_types: Optional[List[str]] = NEREL_RELATION_TYPES,
    ):
        """
        Initialize a new :class:`ArtifactsExtractorLLM`.

        :param client: Language model client for generation and validation.
        :param model_name: Model name to use for generation and validation.
        :param do_validation: Whether to perform additional LLM-based validation of artifacts.
        :param language: Output text language.
        :param entity_types: List of entity types to guide extraction prompts.
        :param relation_types: List of relation types to guide extraction prompts.
        """
        _PROMPTS = ["artifact_extraction", "artifact_validation"]
        super().__init__(prompts=_PROMPTS)

        self.llm = llm
        self.do_validation = do_validation
        self.language = language if language else Settings.language
        self.entity_types = ", ".join(entity_types) if entity_types else None
        self.relation_types = ", ".join(relation_types) if relation_types else None

    async def extract(
        self,
        chunks: List[Chunk],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from a collection of chunks.

        Steps:
          1) Batch-render the extraction prompt with `context=<chunk_texts>`,
          2) Generate structured artifacts per chunk,
          3) Optionally validate artifacts against the original context,
          4) Convert artifacts into Entity/Relation objects.

        :param chunks: Iterable of Chunk objects.
        :return: (entities, relations) extracted from all chunks.
        """

        entities_result: List[Entity] = []
        relations_result: List[Relation] = []

        context: List[str] = [chunk.content for chunk in chunks]

        extraction_instruction: RAGUInstruction = self.get_prompt("artifact_extraction")
        extraction_conversations: List[ChatMessages] = render(
            extraction_instruction.messages,
            context=context,
            language=self.language,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
        )

        result_list = await self.llm.batch_chat_completion( # type: ignore
            [c.to_openai() for c in extraction_conversations],
            output_schema=extraction_instruction.pydantic_model or str, # type: ignore
            desc="Extracting a knowledge graph from chunks",
        )
        result_list = cast(list[BaseModel] | list[str], result_list)
        
        if self.do_validation:
            validation_instruction: RAGUInstruction = self.get_prompt("artifact_validation")

            validation_conversations: List[ChatMessages] = render(
                validation_instruction.messages,
                artifacts=result_list,
                context=context,
                entity_types=self.entity_types,
                relation_types=self.relation_types,
                language=self.language,
            )
            
            result_list = await self.llm.batch_chat_completion( # type: ignore
                [c.to_openai() for c in validation_conversations],
                output_schema=validation_instruction.pydantic_model or str, # type: ignore
                desc="Validation of extracted artifacts",
            )
            result_list = cast(list[BaseModel] | list[str], result_list)

        for artifacts, chunk in zip(result_list, chunks):
            if not hasattr(artifacts, "model_dump"):
                assert False  # what should happen here?
                # continue

            current_chunk_entities: List[Entity] = []

            # Parse entities
            for result in artifacts.model_dump().get("entities", []): # type: ignore
                if result is None:
                    continue
                entity = Entity(
                    entity_name=result.get("entity_name", ""), # type: ignore
                    entity_type=result.get("entity_type", "UNKNOWN"), # type: ignore
                    description=result.get("description", ""), # type: ignore
                    source_chunk_id=[chunk.id],
                    documents_id=[],
                    clusters=[],
                )
                current_chunk_entities.append(entity)

            entities_result.extend(current_chunk_entities)

            # Parse relations
            for result in artifacts.model_dump().get("relations", []): # type: ignore
                if result is None:
                    continue

                subject_name = cast(str, result.get("source_entity", "")) # type: ignore
                object_name = cast(str, result.get("target_entity", "")) # type: ignore

                if not (subject_name and object_name):
                    continue

                subject_entity = next(
                    (e for e in current_chunk_entities if e.entity_name == subject_name),
                    None,
                )
                object_entity = next(
                    (e for e in current_chunk_entities if e.entity_name == object_name),
                    None,
                )
                assert subject_entity
                assert object_entity
                
                assert subject_entity.id and object_entity.id, (
                    'On error here, decide what to do if .id is None,'
                    'but .subject_id and .object_id cannot be None'
                )

                if subject_entity and object_entity:
                    relation = Relation(
                        subject_name=subject_name,
                        object_name=object_name,
                        subject_id=subject_entity.id,
                        object_id=object_entity.id,
                        # TODO use default field values here to ensure correct types
                        relation_type=cast(str, result.get("relation_type", "UNKNOWN")), # type: ignore
                        description=cast(str, result.get("description", "")), # type: ignore
                        relation_strength=cast(float, result.get("relation_strength", 1.0)), # type: ignore
                        source_chunk_id=[chunk.id],
                    )
                    relations_result.append(relation)

        return entities_result, relations_result
