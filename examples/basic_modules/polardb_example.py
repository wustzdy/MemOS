import json
import os
import sys

from memos.configs.graph_db import GraphDBConfigFactory
from memos.graph_dbs.factory import GraphStoreFactory


src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, src_path)


def get_polar_db():
    config = GraphDBConfigFactory(
        backend="polardb",
        config={
            "host": os.getenv("POLAR_DB_HOST", "xxxxx"),
            "port": int(os.getenv("POLAR_DB_PORT", "5432")),
            "user": os.getenv("POLAR_DB_USER", "xxxxx"),
            "password": os.getenv("POLAR_DB_PASSWORD", "xxx"),
            "db_name": os.getenv("POLAR_DB_DB_NAME", "xxx"),
            "user_name": os.getenv("POLARDB_USER_NAME", "xxx"),
            "maxconn": int(os.getenv("POLARDB_POOL_MAX_CONN", "100")),
            "use_multi_db": os.getenv("POLARDB_USE_MULTI_DB", "True").lower()
            == "true",  # 设置为True 不添加user_name过滤条件
            "auto_create": True,
            "embedding_dimension": 1024,
        },
    )
    graph = GraphStoreFactory.from_config(config)
    return graph


def test_search_by_embedding(
    graph,
    vector: list[float],
    user_name: str | None = None,
    filter: dict | None = None,
    knowledgebase_ids: list[str] | None = None,
):
    """Test search_by_embedding function."""
    # Query search_by_embedding
    nodes = graph.search_by_embedding(
        vector=vector,
        top_k=100,
        user_name=user_name,
        filter=filter,
        knowledgebase_ids=knowledgebase_ids,
    )
    print(f"test_search_by_embedding: nodes count: {len(nodes)}")
    for node_i in nodes:
        print(f"Search result id: {node_i['id']}, score: {node_i.get('score', 'N/A')}")


def test_get_node(graph, node_id: str, user_name: str | None = None):
    """Test get_node function - query single node."""
    detail = graph.get_node(id=node_id, user_name=user_name)
    print(f"test_get_node: {detail}")


def test_get_nodes(graph, ids: list[str], user_name: str | None = None):
    """Test get_nodes function - query multiple nodes."""
    detail_list = graph.get_nodes(ids=ids, user_name=user_name)
    print(f"test_get_nodes: count: {len(detail_list)}")
    print(f"test_get_nodes: {detail_list}")


def test_update_node(graph, node_id: str, fields: dict, user_name: str | None = None):
    """Test update_node function."""
    result = graph.update_node(id=node_id, fields=fields, user_name=user_name)
    print(f"test_update_node: {result}")


def test_get_memory_count(graph, scope: str, user_name: str | None = None):
    """Test get_memory_count function."""
    count = graph.get_memory_count(scope, user_name)
    print(f"test_get_memory_count: {count}")


def test_node_not_exist(graph, scope: str, user_name: str | None = None):
    """Test node_not_exist function - check if node exists."""
    is_node_exist = graph.node_not_exist(scope, user_name)
    print(f"test_node_not_exist: {is_node_exist}")


def test_remove_oldest_memory(graph, scope: str, skip_count: int, user_name: str | None = None):
    """Test remove_oldest_memory function - remove oldest memory after skipping."""
    result = graph.remove_oldest_memory(scope, skip_count, user_name)
    print(f"test_remove_oldest_memory: {result}")


def test_delete_node(graph, node_id: str, user_name: str | None = None):
    """Test delete_node function."""
    is_node_deleted = graph.delete_node(id=node_id, user_name=user_name)
    print(f"test_delete_node: {is_node_deleted}")
    """
    
    # detail_list = graph.get_nodes(ids=ids,user_name='memos7a9f9fbbb61c412f94f77fbaa8103c35')
    # print("1111多个node:", len(detail_list))
    # #
    # print("多个node:", detail_list)

    # 4，更新 update_node
    # graph.update_node(id="000009999ef-926f-42e2-b7b5-0224daf0abcd", fields={"name": "new_name"})

    # 4，查询 get_memory_count
    # count = graph.get_memory_count('UserMemory','memos07ba3d044650474c839e721f3a69d38a')
    # print("user count:", count)
    # #
    # # 4，判断node是否存在 node_not_exist 1代表存在，
    # isNodeExist = graph.node_not_exist('UserMemory', 'memos07ba3d044650474c839e721f3a69d38a')
    # print("user isNodeExist:", isNodeExist)
    #
    # # 6,删除跳过多少行之后的数据remove_oldest_memory
    # remove_oldest_memory = graph.remove_oldest_memory('UserMemory', 2,'memos07ba3d044650474c839e721f3a69d38a')
    # print("user remove_oldest_memory:", remove_oldest_memory)

    # 7，更新 update_node
    # isNodeExist = graph.update_node(id="bb079c5b-1937-4125-a9e5-55d4abe6c95d", fields={"status": "inactived","tags": ["yoga", "travel11111111", "local studios5667888"]})
    # print("user update_node:", isNodeExist)

    # 8，删除 delete_node
    # isNodeDeleted = graph.delete_node(id="bb079c5b-1937-4125-a9e5-55d4abe6c95d", user_name='memosbfb3fb32032b4077a641404dc48739cd')
    # print("user isNodeDeleted:", isNodeDeleted)
    """


def add_edge(
    db_name: str,
    source_id: str,
    target_id: str,
    edge_type: str = "Memory",
    user_name: str | None = None,
):
    graph = get_polar_db(db_name)
    graph.add_edge(source_id, target_id, edge_type, user_name)


def edge_exists(
    db_name: str,
    source_id: str,
    target_id: str,
    type: str = "Memory",
    direction: str = "OUTGOING",
    user_name: str | None = None,
):
    graph = get_polar_db(db_name)
    is_edge_exists = graph.edge_exists(
        source_id=source_id,
        target_id=target_id,
        type=type,
        user_name=user_name,
        direction=direction,
    )
    print("edge_exists:", is_edge_exists)


def get_children_with_embeddings(db_name: str, id: str, user_name: str | None = None):
    graph = get_polar_db(db_name)
    children = graph.get_children_with_embeddings(id=id, user_name=user_name)
    print("get_children_with_embedding:", children)


def get_subgraph(center_id, depth, center_status, user_name):
    graph = get_polar_db()
    subgraph = graph.get_subgraph(center_id, depth, center_status, user_name)

    json_str1 = json.dumps(subgraph, ensure_ascii=False, indent=2)
    print(json_str1)


def convert_graph_edges(core_node: dict) -> dict:
    import copy

    data = copy.deepcopy(core_node)

    id_map = {}

    core_node = data.get("core_node", {})
    core_meta = core_node.get("metadata", {})
    if "graph_id" in core_meta and "id" in core_node:
        id_map[core_meta["graph_id"]] = core_node["id"]

    for neighbor in data.get("neighbors", []):
        n_meta = neighbor.get("metadata", {})
        if "graph_id" in n_meta and "id" in neighbor:
            id_map[n_meta["graph_id"]] = neighbor["id"]

    for edge in data.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")

        if src in id_map:
            edge["source"] = id_map[src]
        if tgt in id_map:
            edge["target"] = id_map[tgt]

    return data


def get_grouped_counts(db_name, user_name):
    graph = get_polar_db(db_name)
    grouped_counts = graph.get_grouped_counts(
        group_fields=["status"],
        where_clause="user_name = %s",
        params=[user_name],
        user_name=user_name,
    )
    grouped_counts = graph.get_grouped_counts1(
        group_fields=["status"], params=[user_name], user_name=user_name
    )
    print("get_grouped_counts:", grouped_counts)


def export_graph(
    graph, include_embedding=False, user_name=None, user_id=None, page=1, page_size=10, filter=None
):
    export_graphlist = graph.export_graph(
        include_embedding=include_embedding,
        user_name=user_name,
        page=page,
        page_size=page_size,
        filter=filter,
    )

    """
    # export_graphlist = graph.export_graph(include_embedding=include_embedding, user_name=user_name,filter=filter)
    """
    json_str1 = json.dumps(export_graphlist, ensure_ascii=False, indent=2)
    print("export_graph:", json_str1)


def get_structure_optimization_candidates(db_name, scope, include_embedding, user_name):
    graph = get_polar_db(db_name)
    candidates = graph.get_structure_optimization_candidates(
        scope=scope, include_embedding=include_embedding, user_name=user_name
    )
    print("get_structure_optimization_candidates:", candidates)


def test_get_all_memory_items(
    graph,
    scope: str,
    include_embedding: bool,
    user_name: str,
    filter: dict | None = None,
    knowledgebase_ids: list | None = None,
):
    """Test get_all_memory_items function."""
    memory_items = graph.get_all_memory_items(
        scope=scope,
        include_embedding=include_embedding,
        user_name=user_name,
        filter=filter,
        knowledgebase_ids=knowledgebase_ids,
    )
    print(f"test_get_all_memory_items: count: {len(memory_items)}")


def get_neighbors_by_tag(db_name, user_name):
    graph = get_polar_db(db_name)
    tags = ["旅游建议", "景点"]
    ids = ["39d12b46-ebe4-4f25-b0b7-1582042049e7"]
    neighbors = graph.get_neighbors_by_tag(tags=tags, exclude_ids=ids, user_name=user_name)
    print("get_neighbors_by_tag:", neighbors)


def get_edges(
    db_name: str, id: str, type: str, direction: str, user_name: str | None = None
) -> None:
    graph = get_polar_db(db_name)
    edges = graph.get_edges(id=id, type=type, direction=direction, user_name=user_name)
    print("get_edges:", edges)


def test_get_by_metadata(
    graph,
    filters: list[dict],
    user_name: str,
    filter: dict | None = None,
    knowledgebase_ids: list | None = None,
):
    """Test get_by_metadata function."""
    ids = graph.get_by_metadata(
        filters=filters, user_name=user_name, filter=filter, knowledgebase_ids=knowledgebase_ids
    )
    """
     # ids = graph.get_by_metadata(filter=filter)
    # ids = graph.get_by_metadata(filter=filter)
    """
    print(f"test_get_by_metadata: count: {len(ids)}")
    print(f"test_get_by_metadata: {ids}")


def test_delete_node_by_prams(
    graph,
    memory_ids,
    file_ids,
    filter: dict | None = None,
    writable_cube_ids: list[str] = [] | None,
    batch_size: int = 1000,
):
    """Test delete_node_by_prams function.
    # deleted_count = graph.delete_node_by_prams(memory_ids=memory_ids)
    # deleted_count = graph.delete_node_by_prams(file_ids=file_ids,writable_cube_ids=writable_cube_ids)
    # deleted_count = graph.delete_node_by_prams(filter=filter, writable_cube_ids=writable_cube_ids)
    """

    deleted_count = graph.delete_node_by_prams(filter=filter)
    print(f"test_delete_node_by_prams: deleted {deleted_count} nodes")


def get_user_names_by_memory_ids(graph, memory_ids):
    user_names = graph.get_user_names_by_memory_ids(memory_ids)
    print(f"get_user_names_by_memory_ids: {user_names}")


if __name__ == "__main__":
    # Example vector for testing
    vector = [
        -0.0019477961,
        -0.026848448,
        -0.04810664,
        0.010315357,
        -0.051453188,
        -0.054609593,
        0.06685492,
        -0.007030605,
        0.0070353583,
        -0.0061701997,
        -0.027437897,
        -0.001578202,
        -0.022608219,
        0.006731127,
        -0.018396512,
        0.046357308,
        -0.0048249247,
        -0.010058661,
        -7.145286e-05,
        -0.029871752,
        0.017445788,
        0.0063983733,
        0.0103724,
        0.011456226,
        0.008456691,
        0.011342139,
        -0.049361598,
        -0.009065154,
        -0.027647058,
        0.020079294,
        0.010762197,
        -0.0033346647,
        0.017835584,
        -0.02044057,
        -0.03804798,
        -0.015297151,
        -0.028521724,
        -0.029662592,
        -0.025555464,
        -0.0070401123,
        -0.03483453,
        0.02226596,
        0.05221377,
        -0.026772391,
        0.011190023,
        -0.019157091,
        0.012188283,
        -0.039131805,
        0.0041665486,
        -0.020231409,
        -0.057195563,
        0.024452625,
        0.009616574,
        -0.03800995,
        0.028673839,
        0.022931466,
        -0.08084958,
        0.019176105,
        -0.024471639,
        -0.048677076,
        -0.0013096224,
        0.039930414,
        -0.035614125,
        0.022456104,
        0.019109555,
        0.061112545,
        0.029111173,
        0.067729585,
        -0.04228821,
        -0.049361598,
        -0.029472448,
        0.057157535,
        -0.030898534,
        -0.013918601,
        0.0029971579,
        0.032343633,
        -0.004192693,
        -0.004646664,
        -0.058678694,
        0.013072456,
        -0.017597903,
        -0.050160203,
        -0.0020095932,
        0.037211344,
        -0.06457318,
        0.021258192,
        -0.010828747,
        0.03394085,
        0.011893558,
        0.046471395,
        -0.029586535,
        -0.04803058,
        0.0510729,
        -0.048639044,
        0.006978315,
        -0.012948862,
        -0.006873735,
        0.016647179,
        0.06693098,
        0.026278015,
        0.012207298,
        0.0140897315,
        -0.043162875,
        -0.010515009,
        0.020497613,
        0.0010392603,
        0.09522453,
        0.035481025,
        0.032172505,
        -0.014042195,
        0.0045159394,
        0.013842544,
        0.023197668,
        0.02019338,
        -0.009454952,
        -0.030860504,
        -0.009488227,
        -0.035861313,
        0.048791163,
        0.008741909,
        0.005338316,
        0.03664091,
        0.06613237,
        -0.0071066627,
        -0.0146696735,
        -0.0047060843,
        -0.020079294,
        0.029871752,
        0.035956386,
        -0.0030969838,
        0.032495752,
        0.021999756,
        -0.013215065,
        -0.040843107,
        -0.03122178,
        0.0038884617,
        0.036203574,
        0.03877053,
        -0.04186989,
        -0.06263371,
        0.038409255,
        -0.060656197,
        -0.041831862,
        -0.054761708,
        0.032476734,
        -0.018567642,
        -0.022494132,
        0.057271622,
        0.0048439396,
        0.0027309551,
        -0.003733969,
        0.003981157,
        0.020478597,
        -0.02040254,
        0.028407637,
        -0.08145804,
        -0.016352454,
        -0.0050626057,
        0.0068119382,
        -0.004727476,
        -0.0061844606,
        -0.008894024,
        -0.038846586,
        0.019328222,
        0.03152601,
        -0.014631644,
        0.035576098,
        0.0008027677,
        0.020478597,
        -0.05841249,
        0.010705153,
        -0.026962535,
        0.007206489,
        0.035442997,
        0.018082773,
        0.031164737,
        0.09172586,
        0.016409498,
        0.01285379,
        -0.0098875305,
        0.016960919,
        -0.04107128,
        -0.03146897,
        -0.042402297,
        0.007177967,
        0.043809365,
        -0.04000647,
        -0.057271622,
        -0.0053193015,
        0.084804595,
        -0.037933894,
        -0.04711789,
        0.034092966,
        0.018615179,
        -0.0013428978,
        -0.01964196,
        0.0030660853,
        0.003370317,
        -0.003158781,
        -0.014897847,
        0.015126021,
        0.027076622,
        -0.0014676803,
        -0.0041689253,
        -0.020782828,
        0.00020143467,
        0.009516749,
        -0.02146735,
        0.027418884,
        -0.00058410113,
        0.020725787,
        0.01671373,
        -0.019870134,
        0.000821188,
        -0.059058983,
        -0.036241602,
        0.004789273,
        0.035214823,
        0.012387935,
        -0.007957561,
        -0.03260984,
        0.00573762,
        -0.049361598,
        -0.03470143,
        0.029681606,
        -0.04339105,
        0.00075226044,
        0.054115217,
        0.04228821,
        -0.071456425,
        -0.025726594,
        0.040957194,
        -0.02129622,
        -0.016380977,
        0.028559752,
        -0.015268629,
        -0.022741321,
        0.0102297915,
        0.014470021,
        -0.014403471,
        -0.02036451,
        0.007658083,
        -0.01660915,
        -0.004924751,
        -0.03342746,
        -0.05723359,
        0.013119993,
        0.00959756,
        -0.02112509,
        0.0013571586,
        0.018938424,
        -0.0071874745,
        -0.04803058,
        0.009706893,
        -0.0017243759,
        -0.025175175,
        0.043581195,
        -0.008684865,
        0.00899385,
        0.00844243,
        0.04213609,
        0.048715103,
        0.016675701,
        0.018320454,
        -0.014308398,
        -0.008580285,
        -0.020953959,
        -0.04373331,
        0.0132721085,
        0.044189658,
        -0.019718017,
        0.0017944918,
        0.008390141,
        -0.019794077,
        -0.0020464337,
        0.021543408,
        -0.039588153,
        -0.03215349,
        0.07130431,
        0.014365441,
        -0.014156282,
        0.020212395,
        0.045824904,
        -0.031145722,
        0.039854355,
        -0.022456104,
        -0.040843107,
        -0.03186827,
        -0.0138615575,
        0.0019953323,
        -0.0078101987,
        0.026487174,
        0.052708145,
        0.002968636,
        -0.057347678,
        0.025346305,
        0.014308398,
        -0.16565417,
        -0.0015734484,
        -0.013709442,
        -0.017607411,
        0.0056805764,
        -0.013243587,
        -0.058602635,
        -0.025631523,
        -0.054229304,
        0.04259244,
        -0.0075630103,
        -0.050540496,
        -0.018662715,
        -0.0036270125,
        -0.001971564,
        -0.013871064,
        -0.0016411875,
        -0.0016293034,
        -0.035024676,
        -0.029415404,
        -0.005324055,
        -0.0565871,
        0.044493888,
        0.013015413,
        0.0020523756,
        -0.0124544855,
        0.025764624,
        0.017303178,
        -0.022132857,
        0.0207448,
        0.0035842299,
        -0.03333239,
        -0.006792924,
        0.0017576512,
        -0.02315964,
        0.039512094,
        -0.023939233,
        0.031830244,
        0.008913038,
        -0.02681042,
        0.024357552,
        0.039093778,
        -0.006488692,
        0.05689133,
        0.047954526,
        -0.015544339,
        -0.0065219672,
        0.014802774,
        -0.039512094,
        0.009483473,
        -0.008204749,
        0.003443998,
        -0.025194189,
        -0.082066506,
        -0.04373331,
        -0.00040910847,
        0.043429077,
        0.04856299,
        0.0027285782,
        0.052708145,
        -0.030366128,
        0.0016055354,
        0.0031041142,
        -0.010980863,
        0.007225503,
        -0.031297836,
        0.07932842,
        -0.013985151,
        0.042478353,
        -0.039550122,
        0.060846344,
        -0.023825146,
        0.018472569,
        0.0031207518,
        0.0073966337,
        -0.0051101423,
        -0.06966906,
        -0.026943522,
        0.023939233,
        -0.09864713,
        -0.032343633,
        -0.008689619,
        0.015782021,
        0.010410429,
        -0.014213325,
        -0.0052337362,
        -0.006712112,
        -0.027000565,
        0.018662715,
        0.22482724,
        0.022303987,
        -0.018348975,
        -0.0056995912,
        0.014023181,
        -0.019584917,
        0.0018123179,
        0.040957194,
        -0.013899586,
        -0.00408336,
        0.040234644,
        0.020117322,
        0.01505947,
        -0.04034873,
        0.0017921149,
        0.037344445,
        -0.022874422,
        3.71748e-05,
        0.03456833,
        -0.02243709,
        0.038656443,
        0.017084513,
        -0.0075202277,
        0.0077911844,
        -0.026544217,
        -0.02549842,
        -0.010591066,
        0.0021557668,
        -0.013338659,
        0.031088678,
        -0.051567275,
        0.012929848,
        0.029453434,
        -0.051567275,
        -0.0027784912,
        0.032723922,
        -0.010505501,
        0.00942643,
        -0.01936625,
        0.040576905,
        0.05586455,
        -0.03318027,
        -0.02922526,
        0.011247066,
        -0.020877901,
        0.012616109,
        0.010419936,
        -0.029434418,
        -0.020174365,
        -0.028217493,
        -0.027799172,
        -0.017540859,
        0.034815516,
        -0.024870943,
        0.0342641,
        -0.004991302,
        -0.019832104,
        0.011503762,
        -0.021106075,
        0.015002427,
        -0.04000647,
        0.020782828,
        -0.04624322,
        0.016428513,
        -0.015734484,
        -0.036032446,
        0.057499796,
        -0.00017231875,
        0.019242655,
        0.036412735,
        0.021144105,
        0.025403349,
        0.023349784,
        -0.02388219,
        0.02825552,
        -0.033997893,
        0.04194595,
        0.00067917357,
        0.019138077,
        -0.0248139,
        -0.022494132,
        -0.011275588,
        0.011731936,
        0.03897969,
        0.047193944,
        -0.013376689,
        -0.03359859,
        0.068147905,
        0.024832914,
        -0.026163928,
        0.003921737,
        0.0028640565,
        -0.041983977,
        -0.0019477961,
        0.0069878222,
        0.007639068,
        -0.016799295,
        -0.0563209,
        -0.04297273,
        0.008176227,
        -0.038542356,
        -0.01885286,
        -0.011446718,
        -0.027437897,
        0.00046734032,
        -0.005247997,
        -0.011655877,
        0.016998947,
        -0.021942712,
        -0.02315964,
        -0.012283356,
        0.054875795,
        -0.01895744,
        -0.028597781,
        0.0041427803,
        0.036165547,
        0.03279998,
        0.006303301,
        -0.017017962,
        0.02112509,
        -0.048639044,
        0.047270004,
        -0.014451006,
        -0.003446375,
        0.010914313,
        0.025441376,
        -0.021695524,
        -0.037002183,
        -0.026278015,
        0.022931466,
        0.02422445,
        -0.037610646,
        -0.004447012,
        0.00083841983,
        0.043086816,
        0.0031207518,
        0.007125677,
        -0.050806697,
        0.03456833,
        0.0010850138,
        -0.011171008,
        -0.000560333,
        0.004815418,
        -0.019385265,
        0.00081940537,
        0.044417832,
        -0.015316166,
        -0.00047655046,
        -0.0062462576,
        0.0003437462,
        0.009744923,
        -0.0008574343,
        0.007981329,
        -0.007929039,
        0.0065124603,
        -0.026582247,
        0.011731936,
        0.025270248,
        0.028921027,
        -0.03920786,
        0.035195805,
        -9.826625e-05,
        -0.044341773,
        -0.000839014,
        0.023425842,
        0.03673598,
        0.0005053693,
        0.053126466,
        -0.023501901,
        -0.006122663,
        -0.023102596,
        -0.023577958,
        -0.0149073545,
        0.032780968,
        0.002111796,
        -0.00079563726,
        -0.042098064,
        -0.043771338,
        0.006331823,
        -0.00016979339,
        -0.005804171,
        -0.007715126,
        0.0011515645,
        0.028711868,
        -0.009935067,
        0.021239176,
        0.02133425,
        -0.022741321,
        -0.01967999,
        -0.02298851,
        -0.0029662591,
        0.11020794,
        -0.015829556,
        0.0018028106,
        0.02312161,
        -0.00806214,
        0.033484504,
        0.02643013,
        -0.0015033325,
        -0.066398576,
        0.010429444,
        -0.025061088,
        0.046205193,
        0.018358482,
        -0.03070839,
        0.0036555342,
        0.024794886,
        -0.0067691556,
        -0.03690711,
        -0.012292863,
        0.0017766657,
        0.036089487,
        -0.04669957,
        -0.009345618,
        0.009944574,
        -0.03901772,
        0.035347924,
        0.0068452135,
        -0.0048415624,
        0.004599128,
        0.035956386,
        -0.046433367,
        0.037230358,
        -0.0171986,
        -0.05875475,
        0.00063401414,
        0.033921838,
        0.02591674,
        -0.054533534,
        0.0096498495,
        -0.008123938,
        0.015297151,
        0.023749089,
        -0.027114652,
        0.0220568,
        0.0076010395,
        0.00861356,
        0.029168217,
        0.015572861,
        -0.031487983,
        0.018073266,
        -0.011836515,
        0.024129378,
        0.05909701,
        0.020953959,
        0.023768103,
        -0.01505947,
        0.037648674,
        0.012654138,
        0.004321041,
        -0.016247876,
        0.00374823,
        0.006303301,
        0.016742252,
        0.03304717,
        0.0068975035,
        0.0186437,
        0.02315964,
        0.002783245,
        -0.041755803,
        -0.0021842886,
        0.032571808,
        0.010419936,
        -0.0071351845,
        -0.04882919,
        -0.032933082,
        -0.0023316508,
        -0.097582325,
        -0.028921027,
        0.014080225,
        0.01075269,
        0.00048189829,
        -0.040196616,
        -0.0387325,
        0.022627234,
        -0.0115132695,
        0.010543531,
        0.022380047,
        -0.005295533,
        0.021581437,
        -0.06727324,
        0.03376972,
        0.024585726,
        0.03291407,
        0.006365098,
        -0.038333196,
        0.050160203,
        0.007049619,
        0.012235819,
        -0.031335868,
        0.003959766,
        0.015496803,
        -0.028217493,
        0.035994414,
        -0.012083704,
        -0.0042307223,
        -0.035024676,
        0.025403349,
        -0.03690711,
        -0.023958247,
        -0.0015211586,
        -0.0014962021,
        0.006474431,
        -0.03405494,
        0.008480459,
        -0.0115132695,
        -0.021106075,
        -0.008585039,
        0.039512094,
        -0.023768103,
        -0.00596104,
        -0.027590014,
        -0.011494255,
        0.021182133,
        0.008903531,
        0.024072334,
        0.040690992,
        -0.018653207,
        0.027970303,
        -0.009369386,
        -0.005371591,
        0.02905413,
        -0.027666071,
        -0.011551298,
        -0.0016839701,
        0.004848693,
        -0.009683125,
        0.026487174,
        -0.032362647,
        0.004061969,
        0.00468707,
        -0.048258755,
        0.0049390118,
        0.020972975,
        -0.010923821,
        0.022151873,
        0.053544782,
        -0.06263371,
        0.01795918,
        0.03165911,
        -0.0023720567,
        -0.028065376,
        0.056282867,
        -0.01871025,
        0.01764544,
        0.0034035924,
        0.012226312,
        0.0037268386,
        -0.027304797,
        -0.01113298,
        0.013215065,
        -0.013205558,
        -0.035880327,
        -0.008085908,
        0.0582984,
        -0.04114734,
        -0.041451573,
        0.036888096,
        0.094996355,
        0.011846023,
        -0.019157091,
        0.050806697,
        -0.031202765,
        -0.016666194,
        -0.013005906,
        0.040919166,
        -0.008604053,
        -0.05837446,
        0.0042521134,
        -0.033579577,
        -0.041337486,
        -0.028331578,
        -0.008965328,
        -0.034758475,
        -0.059933648,
        0.06590419,
        0.001622173,
        0.036469776,
        0.06316611,
        -0.06434501,
        -0.026068855,
        -0.006113156,
        -0.014650659,
        -0.010277328,
        -0.035157777,
        0.0014581732,
        -0.069593005,
        -0.011323124,
        -0.01681831,
        0.012720688,
        -0.049399626,
        -0.015943643,
        -0.014631644,
        -0.0015888977,
        -0.13074358,
        0.014365441,
        0.023787117,
        0.0103533855,
        -0.014412978,
        0.02753297,
        -0.013966138,
        -0.06776761,
        0.0100016175,
        -0.06936483,
        0.007729387,
        -0.009578546,
        0.041033253,
        -0.039055746,
        -0.04339105,
        0.021201149,
        0.02295048,
        -0.0023922597,
        0.001446289,
        0.028445665,
        0.018139817,
        -0.030023867,
        0.08419613,
        0.03622259,
        -0.02243709,
        -0.023577958,
        0.00861356,
        0.009255299,
        -0.0065314746,
        0.021885669,
        -0.004007302,
        -0.045064323,
        0.026525203,
        0.0426685,
        0.053544782,
        0.010429444,
        -0.018529613,
        0.0027642304,
        0.053164493,
        0.07278744,
        -0.0021165495,
        0.024699813,
        0.005533214,
        0.00047536206,
        0.01643802,
        0.047460146,
        -0.023235697,
        -0.037439514,
        0.0011165066,
        0.012045675,
        0.021391293,
        0.03300914,
        -0.0221899,
        0.016742252,
        -0.011484748,
        0.0016388107,
        -0.0047060843,
        0.0033845778,
        -0.047155917,
        0.014384456,
        -0.029472448,
        -0.020896915,
        -0.01505947,
        -0.04297273,
        -0.009683125,
        0.041033253,
        -0.027609028,
        0.012492515,
        0.033408444,
        0.04669957,
        0.031640097,
        0.016010195,
        -0.004986548,
        -0.016390484,
        -0.0032419693,
        0.05362084,
        0.047155917,
        0.045406584,
        -0.058260374,
        -0.006964054,
        0.016998947,
        0.020782828,
        -0.00053923886,
        0.04213609,
        0.037610646,
        0.00012196008,
        -0.0067596487,
        -0.02650619,
        -0.012387935,
        2.525361e-06,
        -0.026354073,
        -0.0074061407,
        0.020877901,
        -0.023102596,
        -0.04768832,
        0.023387814,
        -0.04784044,
        0.00897959,
        0.014431993,
        0.0179877,
        -0.008898778,
        -0.04955174,
        -0.0020785206,
        0.019290192,
        -0.01533518,
        -0.053887043,
        0.01412776,
        -0.025897725,
        -0.0059087505,
        -0.01713205,
        0.017864106,
        -0.017360222,
        -0.025840681,
        0.03538595,
        -0.07541144,
        0.0018087527,
        0.01671373,
        0.01561089,
        -0.016342947,
        0.031697143,
        0.020630714,
        0.011332631,
        -0.048524957,
        -0.02660126,
        0.02816045,
        0.05917307,
        0.03202039,
        -0.018824337,
        0.0009091299,
        -0.027114652,
        0.005547475,
        -0.06423092,
        -0.017968686,
        0.015496803,
        0.004958026,
        -0.016276397,
        -0.021239176,
        0.05578849,
        -0.043543164,
        -0.00810017,
        0.021448337,
        -0.042516384,
        0.016875353,
        -0.004199824,
        -0.015582369,
        -0.020820858,
        0.006640808,
        0.021714538,
        -0.011741443,
        0.005770895,
        0.030099925,
        0.0017683469,
        -0.028331578,
        0.08115381,
        -0.008589792,
        0.029719636,
        -0.01775002,
        0.012568573,
        -0.027647058,
        0.049095392,
        -0.0068119382,
        -0.03690711,
        -0.04072902,
        -0.009241038,
        0.051529247,
        -0.032685895,
        -0.006051359,
        -0.0051909536,
        -0.01202666,
        0.01533518,
        -0.008252285,
        0.025137145,
        -0.015867585,
        -0.02367303,
        0.024262479,
        -0.006379359,
        -0.00035978967,
        0.001434405,
        0.055408202,
        0.0045420844,
        0.009416922,
        0.019699004,
        0.012349906,
        0.011456226,
        0.013709442,
        0.011275588,
        -0.0024362307,
        0.052518,
        0.02142932,
        0.0074394164,
        0.072026856,
        -0.034625374,
        0.024889957,
        0.0036032444,
        0.044493888,
        0.010315357,
        -0.013367181,
        -0.00947872,
        -0.029282304,
        0.072597295,
        -0.030156968,
        -0.015202079,
        -0.069707096,
        -0.03373169,
        0.012625616,
        0.014289384,
        0.02825552,
        -0.020421553,
        -0.010077676,
        0.007054373,
        0.027875232,
        -0.0067834165,
        -0.017835584,
        0.027247753,
        0.028806942,
        -0.019290192,
        0.004570606,
        0.00882272,
        0.008917792,
        -0.00036424617,
        0.029605549,
        -0.012948862,
        -0.020839872,
        -0.03935998,
        -0.052441943,
        -0.009830487,
        0.0110284,
        -0.010762197,
        -0.009573792,
        -0.01964196,
        -0.005728113,
        -0.01406121,
        -0.005965794,
        0.008394894,
        -0.0051291566,
        -0.04734606,
        0.0025336798,
        -0.007857735,
        -0.0053478233,
        0.014973905,
        -0.0041285194,
        0.02587871,
        0.018272918,
        -0.0042687515,
    ]
    # Example filter for testing - common filter used by multiple tests
    filter_example = {
        "or": [
            {"id": "27138353-0cb4-44bd-8031-e902b610fa68"},
            {"C": {"contains": "c"}},
            {"tags": {"contains": ["旅游", "广州"]}},
        ]
    }

    filter_example1 = {
        "id": "65c3a65b-78f7-4009-bc92-7ee1981deb3g",
    }

    filter_example2 = {
        "id": "8a3f9960-732c-4737-ade4-ea5536b44da3",
        "and": [
            {"app_id": "123"},
        ],
    }

    filter_example31 = {
        "and": [
            {"id": "65c3a65b-78f7-4009-bc92-7ee1981deb82"},
            {"money": {"gt": 25, "lte": 30}},
        ]
    }
    filter_example3 = {
        "and": [
            {"id": "01d87662-6f82-45d9-acbe-6f1ae660949e"},
            {"file_id": {"like": "a0617d4a6d11ebe4fd13feac0446e6c4"}},
            {"created_at": {"gte": "2025-09-19", "lte": "2025-12-31"}},
            {"confidence": "0.919"},
        ]
    }
    filter_example4 = {"or": [{"Ai": "agent"}, {"learning": {"contains": "学习"}}]}
    filter_example5 = {
        "and": [
            {"created_at": {"lt": "2025-11-29"}},
        ]
    }

    filters_example = [
        {"field": "tags", "op": "contains", "value": "data retrieval"},
    ]

    # Create connection once - shared by all tests
    graph = get_polar_db()
    user_name = "base981b0ad7-77d8-41dd-b9cb-a5741b22c763"
    scope = "UserMemory"
    knowledgebase_ids = ["adimin", "local_test_wwq_10"]

    test_search_by_embedding(graph, vector, user_name, filter_example3, knowledgebase_ids)
    test_get_all_memory_items(
        graph, "LongTermMemory", False, user_name, filter_example3, knowledgebase_ids
    )
    test_get_by_metadata(graph, filters_example, user_name, filter_example3, knowledgebase_ids)

    test_search_by_embedding(graph, vector, user_name, filter_example)

    test_get_node(
        graph,
        '"fb192cff-4da1-402a-9440-46d468f46ff5"',
        '"test_user_456765aax43156789gg921336533nn08khb21122"',
    )

    test_get_nodes(
        graph,
        ["65c3a65b-78f7-4009-bc92-7ee1981deb3a", "65c3a65b-78f7-4009-bc92-7ee1981deb3b"],
        user_name,
    )

    test_update_node(
        graph,
        "bb079c5b-1937-4125-a9e5-55d4abe6c95d",
        {"status": "inactived", "tags": ["yoga", "travel11111111", "local studios5667888"]},
        user_name,
    )

    test_get_memory_count(graph, "UserMemory", user_name)

    test_node_not_exist(graph, "UserMemory", user_name)

    # Test remove_oldest_memory
    test_remove_oldest_memory(graph, "UserMemory", 2, user_name)

    # Test delete_node
    test_delete_node(
        graph, "4050dcd7-e4e8-4fca-81a5-3fb61de2dd28", "memos3c65b19f03de4153b8b30ac2cf1adc6c"
    )

    # Test test_delete_node_by_prams
    memory_ids = ["d8de5abb-57af-4823-9716-fa9ef9095336"]
    file_ids = ["13bf4dad777f89478b7c96cb19442d27", "6860315a37b3180d36fcfe2b0e536118"]
    writable_cube_ids = ["basea68a7f40-d18e-427e-a025-764eef9dd7b6"]

    filter_example5 = {
        "and": [
            {"id": "cddbaef2-336c-4f10-8627-9b5e9ee069e0"},
            {"confidence": 0.99},
            {"type": "fact"},
        ]
    }
    test_delete_node_by_prams(graph, memory_ids, file_ids, filter_example5, writable_cube_ids)

    # Test get_all_memory_itemsn
    test_get_all_memory_items(graph, scope, False, user_name, filter_example)

    # Test get_by_metadata
    test_get_by_metadata(graph, filters_example, user_name, filter_example)

    add_edge(
        db_name="memtensor_memos",
        source_id="13bb9df6-0609-4442-8bed-bba77dadac92",
        target_id="2dd03a5b-5d5f-49c9-9e0a-9a2a2899b98d",
        edge_type="PARENT",
        user_name="memosbfb3fb32032b4077a641404dc48739cd",
    )
    edge_exists(
        db_name="memtensor_memos",
        source_id="13bb9df6-0609-4442-8bed-bba77dadac92",
        target_id="2dd03a5b-5d5f-49c9-9e0a-9a2a2899b98d",
        type="PARENT",
        direction="OUTGOING",
        user_name="memosbfb3fb32032b4077a641404dc48739cd",
    )

    get_children_with_embeddings(
        db_name="memtensor_memos",
        id="13bb9df6-0609-4442-8bed-bba77dadac92",
        user_name="memos07ea708ac7eb412887c5c283f874ea30",
    )

    get_subgraph(
        center_id="503ab3d5-919d-4986-b0f4-15b65c049686",
        depth=1,
        center_status="activated",
        user_name="memos07ea708ac7eb412887c5c283f874ea30",
    )

    get_grouped_counts(db_name="memtensor_memos", user_name="memos07ea708ac7eb412887c5c283f874ea30")
    user_id = "base75fe6c57-c41c-4cd8-b461-ac96c17c241e"
    filter = {
        "and": [
            {"created_at": {"gte": "2025-09-19", "lte": "2025-12-31"}},
            {"memory_type": {"like": "LongTermMemory"}},
            {"memory_type": "LongTermMemory"},
            {"confidence": 0.99},
            {"type": "fact"},
        ]
    }
    export_graph(
        graph,
        include_embedding=False,
        user_name="base981b0ad7-77d8-41dd-b9cb-a5741b22c763",
        page=1,
        page_size=9,
        filter=filter,
    )

    get_structure_optimization_candidates(
        db_name="memtensor_memos",
        scope="UserMemory",
        include_embedding=False,
        user_name="memos8f5530534d9b413bb8981ffc3d48a495",
    )

    memory_ids = ["966cf1d5-44b9-4a20-be74-a392a3eb0a91", "1d8625fe-6d48-4e08-95fe-dc0a044544a0"]
    user_names = get_user_names_by_memory_ids(graph, memory_ids)
