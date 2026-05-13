"""
ncRNA Target Intelligence Platform — Knowledge Graph Layer
"""

import sqlite3
import numpy as np
import pandas as pd
import networkx as nx


def build_knowledge_graph(db_path="ncrna_platform.db") -> nx.Graph:
    conn = sqlite3.connect(db_path)
    G = nx.MultiDiGraph()

    ncrna_df = pd.read_sql_query(
        "SELECT ncrna_id, symbol, biotype, conservation_score FROM ncrna_master",
        conn
    )
    for _, r in ncrna_df.iterrows():
        G.add_node(
            r["ncrna_id"],
            node_type="ncRNA",
            label=r["symbol"],
            biotype=r["biotype"],
            conservation=r["conservation_score"]
        )

    dis_df = pd.read_sql_query(
        "SELECT disease_id, disease_name, stage FROM disease_context",
        conn
    )
    for _, r in dis_df.iterrows():
        G.add_node(
            r["disease_id"],
            node_type="Disease",
            label=r["disease_name"],
            stage=r["stage"]
        )

    ctx_df = pd.read_sql_query(
        "SELECT context_id, cell_type, cell_state, tissue FROM tissue_cell_context",
        conn
    )
    for _, r in ctx_df.iterrows():
        G.add_node(
            r["context_id"],
            node_type="CellType",
            label=f"{r['cell_type']} ({r['cell_state']})",
            tissue=r["tissue"]
        )

    pw_df = pd.read_sql_query(
        "SELECT DISTINCT pathway_id, pathway_name, pathway_db FROM pathway_links",
        conn
    )
    for _, r in pw_df.iterrows():
        G.add_node(
            r["pathway_id"],
            node_type="Pathway",
            label=r["pathway_name"],
            db=r["pathway_db"]
        )

    gene_df = pd.read_sql_query(
        "SELECT DISTINCT hub_gene FROM pathway_links WHERE hub_gene != ''",
        conn
    )
    for _, r in gene_df.iterrows():
        G.add_node(
            r["hub_gene"],
            node_type="Gene",
            label=r["hub_gene"]
        )

    expr_df = pd.read_sql_query("""
        SELECT ncrna_id, disease_id, context_id, log2fc, padj, direction, specificity_tau
        FROM expression_evidence
    """, conn)

    for _, r in expr_df.iterrows():
        G.add_edge(
            r["ncrna_id"],
            r["disease_id"],
            edge_type="expressed_in_disease",
            log2fc=r["log2fc"],
            padj=r["padj"],
            direction=r["direction"],
            weight=abs(r["log2fc"]) * (1 - min(r["padj"], 1))
        )
        G.add_edge(
            r["ncrna_id"],
            r["context_id"],
            edge_type="expressed_in_celltype",
            tau=r["specificity_tau"],
            weight=r["specificity_tau"]
        )

    pert_df = pd.read_sql_query("""
        SELECT ncrna_id, target_gene_effect, perturbation_type, effect_size, confidence
        FROM perturbation_evidence
    """, conn)

    conf_map = {"high": 1.0, "medium": 0.6, "low": 0.3}

    for _, r in pert_df.iterrows():
        tgt_gene = r["target_gene_effect"]
        if tgt_gene not in G.nodes:
            G.add_node(tgt_gene, node_type="Gene", label=tgt_gene)

        G.add_edge(
            r["ncrna_id"],
            tgt_gene,
            edge_type="perturbs_gene",
            perturbation_type=r["perturbation_type"],
            effect_size=r["effect_size"],
            confidence=r["confidence"],
            weight=abs(r["effect_size"]) * conf_map.get(r["confidence"], 0.5)
        )

    pathway_links_df = pd.read_sql_query("""
        SELECT ncrna_id, pathway_id, hub_gene, pearson_r
        FROM pathway_links
    """, conn)

    for _, pl in pathway_links_df.iterrows():
        G.add_edge(
            pl["ncrna_id"],
            pl["pathway_id"],
            edge_type="pathway_member",
            pearson_r=pl["pearson_r"],
            weight=abs(pl["pearson_r"])
        )

        if pl["hub_gene"] and pl["hub_gene"] in G.nodes:
            G.add_edge(
                pl["pathway_id"],
                pl["hub_gene"],
                edge_type="pathway_contains_gene",
                weight=0.5
            )

    clin_df = pd.read_sql_query("""
        SELECT ncrna_id, disease_id, correlation_r, pvalue, biomarker_type
        FROM clinical_links
    """, conn)

    for _, r in clin_df.iterrows():
        G.add_edge(
            r["ncrna_id"],
            r["disease_id"],
            edge_type="clinical_association",
            correlation_r=r["correlation_r"],
            pvalue=r["pvalue"],
            biomarker_type=r["biomarker_type"],
            weight=abs(r["correlation_r"]) * (1 - min(r["pvalue"] * 10, 1))
        )

    conn.close()
    print(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def extract_graph_features(G: nx.Graph, ncrna_ids: list) -> pd.DataFrame:
    G_undirected = G.to_undirected()

    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=200, weight="weight")
    except Exception:
        pagerank = {n: 0 for n in G.nodes}

    try:
        betweenness = nx.betweenness_centrality(G_undirected, normalized=True)
    except Exception:
        betweenness = {n: 0 for n in G.nodes}

    degree_cent = nx.degree_centrality(G_undirected)

    rows = []
    for nid in ncrna_ids:
        if nid not in G:
            continue

        out_edges = list(G.out_edges(nid, data=True))

        n_disease_edges = sum(
            1 for _, _, d in out_edges
            if d.get("edge_type", "").startswith("expressed_in_disease")
        )
        n_gene_edges = sum(
            1 for _, _, d in out_edges
            if d.get("edge_type", "") == "perturbs_gene"
        )
        n_pathway_edges = sum(
            1 for _, _, d in out_edges
            if d.get("edge_type", "") == "pathway_member"
        )
        n_clinical_edges = sum(
            1 for _, _, d in out_edges
            if d.get("edge_type", "") == "clinical_association"
        )

        disease_neighbors = set(
            v for _, v, d in out_edges if G.nodes[v].get("node_type") == "Disease"
        )

        rows.append({
            "ncrna_id": nid,
            "graph_pagerank": pagerank.get(nid, 0),
            "graph_betweenness": betweenness.get(nid, 0),
            "graph_degree_cent": degree_cent.get(nid, 0),
            "graph_n_disease_edges": n_disease_edges,
            "graph_n_gene_edges": n_gene_edges,
            "graph_n_pathway_edges": n_pathway_edges,
            "graph_n_clinical_edges": n_clinical_edges,
            "graph_disease_diversity": len(disease_neighbors),
        })

    return pd.DataFrame(rows)


def predict_novel_associations(G: nx.Graph, ncrna_ids: list, disease_ids: list, top_k: int = 5) -> pd.DataFrame:
    existing_pairs = set()

    for nid in ncrna_ids:
        for _, v, d in G.out_edges(nid, data=True):
            if G.nodes[v].get("node_type") == "Disease":
                existing_pairs.add((nid, v))

    results = []

    for nid in ncrna_ids:
        for did in disease_ids:
            if (nid, did) in existing_pairs:
                continue

            ncrna_nbrs = set(G.successors(nid)) | set(G.predecessors(nid))
            disease_nbrs = set(G.successors(did)) | set(G.predecessors(did))

            union = ncrna_nbrs | disease_nbrs
            intersect = ncrna_nbrs & disease_nbrs

            jaccard = len(intersect) / len(union) if union else 0.0

            aa_score = 0.0
            for s in intersect:
                deg = G.degree(s)
                if deg > 1:
                    aa_score += 1.0 / np.log1p(deg)

            results.append({
                "ncrna_id": nid,
                "disease_id": did,
                "jaccard": round(jaccard, 4),
                "adamic_adar": round(aa_score, 4),
                "shared_nodes": len(intersect),
                "link_score": round((jaccard + aa_score) / 2, 4),
            })

    if not results:
        return pd.DataFrame(columns=[
            "ncrna_id",
            "disease_id",
            "jaccard",
            "adamic_adar",
            "shared_nodes",
            "link_score",
        ])

    df = pd.DataFrame(results)
    return df.sort_values("link_score", ascending=False).head(top_k)


if __name__ == "__main__":
    G = build_knowledge_graph()

    ncrna_ids = [n for n, d in G.nodes(data=True) if d.get("node_type") == "ncRNA"]
    disease_ids = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Disease"]

    gf = extract_graph_features(G, ncrna_ids)
    print(gf.to_string(index=False))

    novel = predict_novel_associations(G, ncrna_ids, disease_ids, top_k=5)
    print(novel.to_string(index=False))