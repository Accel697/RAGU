from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Type, TypeVar

from typing_extensions import override

from ragu.storage.base_storage import BaseGraphStorage, EdgeSpec
from ragu.storage.types import Node, Edge

try:
    import psycopg
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row

    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False

NodeT = TypeVar("NodeT", bound=Node)
EdgeT = TypeVar("EdgeT", bound=Edge)


@dataclass(slots=True)
class AgeNode(Node):
    id: str
    label: str = "Entity"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgeEdge(Edge):
    id: str
    subject_id: str
    object_id: str
    label: str = "RELATED_TO"
    properties: Dict[str, Any] = field(default_factory=dict)


class AgeGraphStorage(BaseGraphStorage[AgeNode, AgeEdge]):

    def __init__(
            self,
            host: str,
            port: int,
            database: str,
            user: str,
            password: str,
            graph_name: str,
            node_cls: Type[AgeNode] = AgeNode,
            edge_cls: Type[AgeEdge] = AgeEdge,
            **kwargs: Any,
    ):
        if not PSYCOPG_AVAILABLE:
            raise ImportError("psycopg3 is required")

        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._graph_name = graph_name
        self._node_cls = node_cls
        self._edge_cls = edge_cls
        self._conn: Optional[AsyncConnection] = None
        self._initialized: bool = False

    async def _connect(self) -> None:
        if self._conn is None or self._conn.closed:
            conninfo = f"host={self._host} port={self._port} dbname={self._database} user={self._user} password={self._password}"
            self._conn = await psycopg.AsyncConnection.connect(conninfo)
            async with self._conn.cursor() as cursor:
                await cursor.execute("CREATE EXTENSION IF NOT EXISTS age")
                await cursor.execute("LOAD 'age'")
                await cursor.execute("SET search_path = ag_catalog, public")
                await self._conn.commit()

    def _escape(self, value: Any) -> str:
        if isinstance(value, str):
            return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif value is None:
            return "null"
        else:
            return str(value)

    def _parse_agtype(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        cleaned = value.split("::")[0]
        try:
            return json.loads(cleaned)
        except:
            return value

    async def _run_cypher(self, query: str, fetch: bool = True) -> List[Dict[str, Any]]:
        if self._conn is None or self._conn.closed:
            raise RuntimeError("Not connected")

        try:
            async with self._conn.cursor(row_factory=dict_row) as cursor:
                wrapped_query = f"SELECT * FROM cypher('{self._graph_name}', $${query}$$) AS (result agtype);"
                await cursor.execute(wrapped_query)

                if fetch:
                    rows = await cursor.fetchall()
                    return [{"result": self._parse_agtype(row["result"])} for row in rows]
                else:
                    await self._conn.commit()
                    return []
        except Exception as e:
            await self._conn.rollback()
            raise RuntimeError(f"AGE query failed: {e}")

    async def _ensure_graph_exists(self) -> None:
        async with self._conn.cursor() as cursor:
            await cursor.execute("SELECT 1 FROM ag_graph WHERE name = %s", (self._graph_name,))
            exists = await cursor.fetchone()
            if not exists:
                await cursor.execute("SELECT create_graph(%s)", (self._graph_name,))
                await self._conn.commit()

    @override
    async def index_start_callback(self) -> None:
        if not self._initialized:
            await self._connect()
            await self._ensure_graph_exists()
            self._initialized = True

    @override
    async def index_done_callback(self) -> None:
        if self._conn:
            await self._conn.commit()

    @override
    async def query_done_callback(self) -> None:
        pass

    @override
    async def get_nodes(self, node_ids: List[str]) -> List[Optional[AgeNode]]:
        nodes = []
        for nid in node_ids:
            query = f"MATCH (n) WHERE n.id = {self._escape(nid)} RETURN n"
            results = await self._run_cypher(query)
            if results and results[0].get("result"):
                data = results[0]["result"]
                props = data.get("properties", data)
                nodes.append(self._node_cls(
                    id=str(props.get("id", "")),
                    label=str(props.get("label", data.get("label", "Entity"))),
                    properties={k: v for k, v in props.items() if k not in ("id", "label")}
                ))
            else:
                nodes.append(None)
        return nodes

    @override
    async def upsert_nodes(self, nodes: Iterable[AgeNode]) -> None:
        for node in nodes:
            parts = [f"n.{k} = {self._escape(v)}" for k, v in node.properties.items()]
            parts.append(f"n.id = {self._escape(node.id)}")
            parts.append(f"n.label = {self._escape(node.label)}")
            query = f"MERGE (n:{node.label} {{id: {self._escape(node.id)}}}) SET {', '.join(parts)}"
            await self._run_cypher(query, fetch=False)

    @override
    async def delete_nodes(self, node_ids: List[str]) -> None:
        if not node_ids:
            return
        ids_str = ", ".join(self._escape(nid) for nid in node_ids)
        query = f"MATCH (n) WHERE n.id IN [{ids_str}] DETACH DELETE n"
        await self._run_cypher(query, fetch=False)

    @override
    async def get_all_nodes(self) -> List[AgeNode]:
        results = await self._run_cypher("MATCH (n) RETURN n")
        nodes = []
        for row in results:
            data = row.get("result", {})
            if data:
                props = data.get("properties", data)
                nodes.append(self._node_cls(
                    id=str(props.get("id", "")),
                    label=str(props.get("label", data.get("label", "Entity"))),
                    properties={k: v for k, v in props.items() if k not in ("id", "label")}
                ))
        return nodes

    @override
    async def get_edges(self, edge_specs: List[EdgeSpec]) -> List[Optional[AgeEdge]]:
        edges = []
        for subject_id, object_id, relation_id in edge_specs:
            if relation_id:
                query = f"MATCH (a)-[r:`{relation_id}`]->(b) WHERE a.id = {self._escape(subject_id)} AND b.id = {self._escape(object_id)} RETURN r"
            else:
                query = f"MATCH (a)-[r]->(b) WHERE a.id = {self._escape(subject_id)} AND b.id = {self._escape(object_id)} RETURN r"
            results = await self._run_cypher(query)
            if results and results[0].get("result"):
                data = results[0]["result"]
                props = data.get("properties", data)
                edges.append(self._edge_cls(
                    id=str(props.get("id", "")),
                    subject_id=subject_id,
                    object_id=object_id,
                    label=str(props.get("label", data.get("label", "RELATED_TO"))),
                    properties={k: v for k, v in props.items() if k not in ("id", "label", "subject_id", "object_id")}
                ))
            else:
                edges.append(None)
        return edges

    @override
    async def upsert_edges(self, edges: Iterable[AgeEdge]) -> None:
        for edge in edges:
            props_parts = [f"id: {self._escape(edge.id)}"] + [f"{k}: {self._escape(v)}" for k, v in
                                                              edge.properties.items()]
            query = f"MATCH (a {{id: {self._escape(edge.subject_id)}}}) MATCH (b {{id: {self._escape(edge.object_id)}}}) MERGE (a)-[r:`{edge.label}` {{{', '.join(props_parts)}}}]-(b)"
            await self._run_cypher(query, fetch=False)

    @override
    async def delete_edges(self, edge_specs: List[EdgeSpec]) -> None:
        for subject_id, object_id, relation_id in edge_specs:
            if relation_id:
                query = f"MATCH (a)-[r:`{relation_id}`]->(b) WHERE a.id = {self._escape(subject_id)} AND b.id = {self._escape(object_id)} DELETE r"
            else:
                query = f"MATCH (a)-[r]->(b) WHERE a.id = {self._escape(subject_id)} AND b.id = {self._escape(object_id)} DELETE r"
            await self._run_cypher(query, fetch=False)

    @override
    async def get_all_edges_for_nodes(self, node_ids: List[str]) -> List[List[AgeEdge]]:
        if not node_ids:
            return []
        grouped = []
        for nid in node_ids:
            edges = []
            seen = set()
            for direction in ["(n)-[r]->(m)", "(m)-[r]->(n)"]:
                query = f"MATCH {direction} WHERE n.id = {self._escape(nid)} RETURN r, m.id as other"
                wrapped_query = f"SELECT * FROM cypher('{self._graph_name}', $${query}$$) AS (r agtype, other agtype);"
                async with self._conn.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(wrapped_query)
                    rows = await cursor.fetchall()
                    for row in rows:
                        data = self._parse_agtype(row.get("r"))
                        if data:
                            props = data.get("properties", data)
                            eid = str(props.get("id", data.get("id", "")))
                            if eid not in seen:
                                seen.add(eid)
                                is_out = direction.startswith("(n)")
                                edges.append(self._edge_cls(
                                    id=eid,
                                    subject_id=nid if is_out else str(row.get("other", "")),
                                    object_id=str(row.get("other", "")) if is_out else nid,
                                    label=str(props.get("label", data.get("label", "RELATED_TO"))),
                                    properties={k: v for k, v in props.items() if k not in ("id", "label")}
                                ))
            grouped.append(edges)
        return grouped

    @override
    async def get_all_edges(self) -> List[AgeEdge]:
        results = await self._run_cypher("MATCH ()-[r]->() RETURN r")
        edges = []
        for row in results:
            data = row.get("result", {})
            if data:
                props = data.get("properties", data)
                edges.append(self._edge_cls(
                    id=str(props.get("id", "")),
                    subject_id=str(props.get("subject_id", "")),
                    object_id=str(props.get("object_id", "")),
                    label=str(props.get("label", data.get("label", "RELATED_TO"))),
                    properties={k: v for k, v in props.items() if k not in ("id", "label", "subject_id", "object_id")}
                ))
        return edges

    @override
    async def edges_degrees(self, edge_specs: List[EdgeSpec]) -> List[int]:
        degrees = []
        for subject_id, object_id, _ in edge_specs:
            query = f"OPTIONAL MATCH (s) WHERE s.id = {self._escape(subject_id)} OPTIONAL MATCH (o) WHERE o.id = {self._escape(object_id)} RETURN (COUNT{{(s)--()}} + COUNT{{(o)--()}}) as d"
            results = await self._run_cypher(query)
            if results and results[0].get("result") is not None:
                val = results[0]["result"]
                degrees.append(int(val) if isinstance(val, (int, float, str)) else 0)
            else:
                degrees.append(0)
        return degrees

    async def close(self) -> None:
        if self._conn:
            await self._conn.commit()
            await self._conn.close()
            self._conn = None

    async def __aenter__(self):
        await self.index_start_callback()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.index_done_callback()
        await self.close()