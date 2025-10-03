# Galaxy Mining: Frequent Subgraph Patterns in Galaxy Workflows

## Project Overview

**Galaxy Mining** is a project that applies **frequent subgraph mining** techniques to real-world Galaxy workflow data in order to extract recurrent patterns of tool usage. Galaxy is an open-source platform for building and sharing scientific data analysis workflows[\[1\]](https://en.wikipedia.org/wiki/Galaxy_%28computational_biology%29#:~:text=Galaxy,throughput%5B%2012). Each Galaxy workflow (or user session) can be represented as a graph of tool executions. This project leverages a **neural subgraph matching** approach to find common subgraphs (motifs) across many workflows. By using a pre-trained Graph Neural Network (GNN) model for subgraph pattern mining, the system efficiently discovers frequently occurring tool combinations and sequences. The results are then integrated with a Neo4j Aura graph database for interactive visualization and analysis of both the patterns and their occurrences in actual sessions.

## Features

- **Custom Graph Dataset Loading** - Supports ingesting Galaxy workflow data as a graph dataset. The tool-sequence data (e.g. derived from Galaxy usage logs) is loaded into NetworkX graphs with attributes (e.g. edge weights representing frequency). This allows the pipeline to accept custom or updated datasets (via a pickled graph file) without code changes.
- **Neural Subgraph Pattern Mining** - Utilizes a pre-trained **neural subgraph matcher** (based on the SPMiner algorithm) to mine frequent patterns. The mining algorithm uses a GNN-based decoder to search for subgraphs that occur frequently in the input graph dataset[\[2\]](https://github.com/snap-stanford/neural-subgraph-learning-GNN#:~:text=Frequent%20Subgraph%20Mining). Patterns are identified along with their structural relationships (i.e. the specific tools and directed connections in each pattern). This means the output patterns are not just frequent sets of tools, but actual subgraph structures (including the pattern's edges reflecting tool execution order or dependency).
- **Integration with Neo4j Aura** - Provides integration with Neo4j (AuraDB) for visualizing the discovered patterns and exploring their context. The frequent patterns and (optionally) the Galaxy sessions can be exported to a Neo4j graph database. Neo4j's visualization tools allow interactive inspection, where patterns, tools, and sessions are all represented as nodes and relationships.
- **Pattern-Session Relationship Mapping** - Supports mapping patterns to the Galaxy **sessions/workflows** in which they appear. Each discovered pattern can be linked to one or more session graphs that contain that subgraph. This relationship mapping allows users to query which sessions a given pattern occurred in, or conversely, which patterns were found in a particular session. It provides insight into how common workflows are composed of these building-block motifs.
- **Pattern Co-occurrence Analytics** - Enables analysis of how patterns co-occur with each other across sessions. For example, if certain patterns tend to appear in the same workflow frequently, the system can represent this with a **co-occurrence relationship** between those pattern nodes in Neo4j. These pattern-to-pattern links (optionally weighted by co-occurrence count) help identify higher-level structures or pipelines (e.g. two subgraph patterns that frequently chain together). This feature is useful for exploring relationships between frequent motifs and could aid in recommendations or clustering of workflows.

## Installation & Setup

Follow these steps to set up the project environment and data:

- **Clone the Repository**: Clone the galaxy-mining repository to your local machine.  

- git clone <https://github.com/nathnael-desta/galaxy-mining.git>  
    cd galaxy-mining

- **Create a Python Environment**: It's recommended to use **Conda** (or venv) to manage dependencies. Create a Conda environment with Python 3.11 (the project has been tested on Python 3.11). For example:  

- conda create -n galaxy-mining-env python=3.11 -y  
    conda activate galaxy-mining-env

- **Install Dependencies**: Install the required Python libraries. You can use pip to install the dependencies. This project relies on PyTorch (for the GNN), PyTorch Geometric, DeepSNAP, NetworkX, and Neo4j Python driver, among others. For example:  

- pip install torch torchvision torchaudio # PyTorch (use appropriate CUDA version if needed)  
    pip install torch-geometric deepsnap networkx==2.8  
    pip install numpy pandas tqdm seaborn matplotlib # common scientific libs  
    pip install neo4j dotenv # for Neo4j integration
- _(If a requirements.txt is provided in the repo, you can alternatively do pip install -r requirements.txt to install all needed packages.)_  
    **Note:** The neural-subgraph-matcher-miner subdirectory contains the GNN model code and may have its own requirements (the above commands cover the major ones). Ensure PyTorch Geometric and DeepSNAP versions are compatible with your PyTorch version.

- **Obtain the Dataset**: The mining process requires a prepared Galaxy workflow graph dataset file (data/aura_tool_graph_attr_weighted.pkl). This file is a pickled directed graph (NetworkX DiGraph) where nodes correspond to Galaxy tools and directed edges represent tool usage transitions in workflows (with edge weights indicating frequency of that transition across sessions).
- **If you have the dataset file provided**: Place aura_tool_graph_attr_weighted.pkl in the data/ folder (or adjust the config to point to its location). This file might be obtained from the project author or generated from Galaxy logs.
- **To prepare the dataset yourself**: If you have access to Galaxy usage data or a Neo4j instance containing the tool usage graph, you can generate the pickle. For example, scripts/export_aura_to_pyg.py demonstrates how to fetch tool nodes and TOOL_CO_OCCURRENCE edges from a Neo4j Aura database and construct the graph. After obtaining a NetworkX graph (e.g., aura_tool_graph.pkl), run scripts/label_pickle.py to add necessary attributes (id, label for nodes and float edge_weight for edges) and save the final aura_tool_graph_attr_weighted.pkl. Ensure the graph is directed and each edge has an edge_weight attribute (the mining code expects this).
- If you do not have the real Galaxy data, you may need to use a placeholder or synthetic dataset. The mining algorithm is general, but results will be meaningful only with real workflow data.
- **Verify Installation (Optional)**: You can run scripts/smoke_env.py to check that all modules import correctly and the model checkpoint is accessible. This will print versions of Torch, NetworkX, and confirm the presence of the pre-trained model file at neural-subgraph-matcher-miner/ckpt/model.pt. The model checkpoint model.pt should be present in that folder (it is provided with the repository) - this is the pre-trained subgraph matching GNN used by the miner.

## Usage

Once the environment is set up and the dataset is ready, you can run the subgraph mining pipeline and explore the results. Below are the typical steps:

- **Run the Pattern Miner**: Use the run_miner.py script to execute the subgraph mining process on the dataset. This will invoke the neural subgraph mining decoder with the configured parameters. By default, the script runs in a fast mode for quick results. You can control the mining profile via an environment variable:

- \# FAST profile (quick test run, default)  
    python scripts/run_miner.py  
    <br/>\# FULL profile (more thorough mining - larger search, more patterns found)  
    MINER_PROFILE=FULL python scripts/run_miner.py
- The script will load the pre-trained model from neural-subgraph-matcher-miner/ckpt/model.pt and the dataset from data/aura_tool_graph_attr_weighted.pkl. It then launches the decoder (subgraph_mining.decoder) to find frequent subgraph patterns. Key parameters like min_pattern_size=3, max_pattern_size=8 (pattern node count limits) and search strategy (greedy) are already set in the script. If using the FAST profile, the mining uses a smaller sample (e.g., 400 neighborhoods, 150 trials) for a quick result; FULL uses larger values (2000 neighborhoods, 1000 trials) for a more exhaustive search.  
    After running, the discovered patterns are saved as a list of NetworkX graphs in a pickle file under results/. By default:

- FAST run outputs to **results/mined_patterns_fast.pkl**
- FULL run outputs to **results/mined_patterns.pkl**  
    The console output will also indicate progress and when the process is complete.
- **View Mining Results**: Once mining is complete, you can inspect the patterns. Each pattern is a subgraph (NetworkX DiGraph) representing a recurring tool sequence structure. To examine results:
- Use the scripts/results_checker.py (if provided) or open a Python shell to load the pickle file and explore. For example:  

- import pickle, networkx as nx  
    patterns = pickle.load(open("results/mined_patterns_fast.pkl", "rb"))  
    print(f"Found {len(patterns)} patterns.")  
    for i, g in enumerate(patterns\[:5\], 1):  
    print(f"Pattern {i}: nodes={g.nodes(data=True)}, edges={g.edges(data=True)}")
- This will print basic info for the first 5 patterns. Each pattern graph's nodes have attributes (like an id or label which should be the tool name) and edges have weights (if applicable). For example, a pattern might show something like "nodes=\['MAFFT', 'Tool_055', 'Diamond'\] and edges=\[('MAFFT', 'Diamond'), ('MAFFT', 'Tool_055'), ('Diamond', 'Tool_055'), ('Tool_055', 'MAFFT')\]", indicating a 3-tool sub-network with specific directed connections.

- You can also generate a prepared Cypher parameter file for Neo4j using scripts/export_patterns_for_aura.py. Running this script will read the mined patterns pickle and produce a file data/patterns_param.cypher.txt containing a Cypher parameter (:param) definition. This file lists all patterns with their details (pattern ID, size, node list, and edge list) in a format ready to be consumed by Neo4j.
- **Load Results into Neo4j**: _This step is optional but recommended for visualization._ If you have a Neo4j Aura instance (or a local Neo4j database) set up, you can load the patterns and the tool graph for exploration:
- **Connect to Neo4j**: Ensure your Neo4j instance is running. If using Aura, have your connection URI, username, and password ready. (You might have a .env file as used by the export script to store these credentials).
- **Load Tool Nodes and Global Graph**: First, create nodes for each unique Galaxy tool and the relationships for tool co-occurrence (transitions). In the data/ folder, load_tools.cypher contains Cypher commands to create all Tool nodes with their id (tool name or identifier). You can copy-paste those MERGE statements into the Neo4j Browser query editor and run them to create the tool nodes. If you also want to visualize the entire tool usage graph, you should create relationships between tools. For example, if you have the aura_tool_graph.edgelist (tab-separated: ToolA, ToolB, weight), you can use Neo4j's LOAD CSV or write a Cypher query to create (:Tool)-\[:TOOL_CO_OCCURRENCE {weight:&lt;count&gt;}\]->(:Tool) relationships for each edge. This step will reproduce the aggregated workflow graph inside Neo4j.
- **Load Patterns**: Now load the mined patterns. Open the patterns_param.cypher.txt file generated earlier and copy its contents into the Neo4j Browser. It defines a parameter \$patterns which is a list of maps describing each pattern. After setting that parameter (you should see a confirmation in Neo4j), run a Cypher query to create Pattern nodes and their connections to Tool nodes. For example:  

- UNWIND \$patterns AS pat  
    CREATE (p:Pattern {id: pat.pid, size: pat.size, edge_count: pat.edge_count})  
    WITH p, pat  
    UNWIND pat.nodes AS toolId  
    MATCH (t:Tool {id: toolId})  
    MERGE (p)-\[:INCLUDES\]->(t);
- This will create a node for each pattern (labeled Pattern with properties like id and size) and connect it to the corresponding Tool nodes it includes. Now each pattern node is linked to the tools that form that pattern. (The pattern's internal edges can be inferred from those tools and possibly from an attribute list; by default we attach tools but not duplicate the pattern's internal tool-to-tool edges in the Neo4j model to keep it simple. The edge_count property on Pattern can be used to know how many connections are in the subgraph.)

- **(Optional) Load Sessions and Map Pattern Occurrence**: If you have the individual session (workflow) data and want to map patterns to sessions, you can create Session nodes and appropriate relationships. For instance, for each workflow session, create a (:Session {id: ...}) node and connect it to the tools or tool-tool edges that occurred in that session. A simpler approach is to link patterns directly to sessions: if a pattern's set of tools and edges is a subgraph of a session, create a relationship like (p:Pattern)-\[:APPEARS_IN\]->(s:Session). You would need to iterate over sessions and patterns to establish these links (this can be done via scripting or Cypher queries if session data is in Neo4j). Once done, you'll have a bipartite mapping of patterns to the sessions containing them.
- **(Optional) Create Pattern Co-occurrence Relationships**: With pattern-to-session mapping in place, you can derive pattern co-occurrence. For example, run a Cypher query to find patterns that share a session:  

- MATCH (p1:Pattern)-\[:APPEARS_IN\]->(s:Session)<-\[:APPEARS_IN\]-(p2:Pattern)  
    WHERE p1 <> p2  
    MERGE (p1)-\[r:CO_OCCURS_WITH\]-(p2)  
    ON CREATE SET r.weight = 1  
    ON MATCH SET r.weight = r.weight + 1;
- This will connect pattern nodes with an undirected relationship CO_OCCURS_WITH and count how many sessions they co-occur in (storing that in r.weight). After this, highly related patterns (that often appear together in the same workflows) will be directly linked in the graph.

- **Example Queries (Neo4j)**: With the data loaded into Neo4j, you can perform various queries to explore and visualize the patterns:
- _Find tools used in a specific session:_  

- MATCH (s:Session {id: "S123"})-\[:CONTAINS\]->(t:Tool)  
    RETURN s, t;
- (This assumes you modeled a CONTAINS relationship from Session to Tool for each tool used in session S123. It will list all tools in that session.)

- _Visualize a particular pattern:_  

- MATCH (p:Pattern {id: "P005"})-\[:INCLUDES\]->(t:Tool)  
    RETURN p, t;
- This retrieves pattern **P005** and all tools that are part of it. In the Neo4j Browser, you will see the Pattern node connected to its tools. You can visually distinguish the pattern by its central Pattern node. _(If you also loaded the global tool graph edges, you could overlay the pattern's edges: those tool nodes will likely be interconnected by TOOL_CO_OCCURRENCE relations which represent how they actually connect in workflows. By focusing on just the pattern's tools, you effectively highlight that subgraph in the global graph.)_

- _Find all patterns present in a given session:_  

- MATCH (s:Session {id: "S123"})<-\[:APPEARS_IN\]-(p:Pattern)  
    RETURN s, p;
- This will return the session node and all Pattern nodes that have an APPEARS_IN relationship to that session (i.e. all patterns found in that workflow). You can add -\[:INCLUDES\]->(t:Tool) in the return path to also see the tools for each pattern.

- _Patterns that co-occur with a given pattern:_  

- MATCH (:Pattern {id:"P001"})-\[:CO_OCCURS_WITH\]-(p2:Pattern)  
    RETURN p2.id, p2.size, p2.edge_count, p2;
- This finds all patterns that are linked to pattern P001 via co-occurrence. The query returns their IDs and sizes; you can also visualize by returning the pattern nodes and relationships. A dense cluster of pattern nodes connected by CO_OCCURS_WITH might indicate a set of motifs that often appear together in workflows.

- _Find patterns sharing a particular tool:_  

- MATCH (t:Tool {id:"FastQC"})<-\[:INCLUDES\]-(p:Pattern)  
    RETURN t, p;
- This will list all patterns that include the tool **FastQC**. This is useful for understanding which common sub-workflows involve a given tool.

These queries are just examples - you can adjust or combine them to answer other questions (such as comparing two patterns' compositions, counting how many sessions a pattern appears in, etc.). Neo4j's visualization can be used to highlight subgraphs: for instance, you can display a session node with its tools and patterns to see how the patterns overlay on the actual tools used.

## Visualization

Using Neo4j Aura (or Neo4j Desktop) to visualize the data greatly helps in interpreting the results:

- **Pattern Graphs**: In Neo4j, each Pattern node connected to its Tool nodes gives a clear picture of the tools involved in that pattern. Because each pattern node has an INCLUDES relationship to multiple tools, you can spot common tools across different patterns (if two pattern nodes share a tool node in the visualization, that means those patterns have a tool in common). The structure of a pattern (i.e., which tools are connected to which within the pattern) can be inferred by considering the subgraph among those tools. If you loaded the global tool network, you can visually inspect how the tools in a pattern are interlinked by actual workflow connections. This effectively highlights the motif within the larger graph. For example, a pattern might form a small loop or a chain among a subset of tools - you would see those tool nodes and the TOOL_CO_OCCURRENCE edges between them in the Neo4j view, centered around the Pattern node.
- **Session Graphs**: If session data is modeled (each session with its tool relationships), you can visualize an entire workflow as a graph of tools (perhaps with a session node connecting to all its tools or as a subgraph of tools connected by execution order). By also displaying the Pattern nodes that appear in that session (as per the APPEARS_IN relations), you can **visually overlay patterns on a session**. In the Neo4j Browser, you might, for instance, see a Session node connected to Pattern P001 and Pattern P005, indicating those patterns exist in that workflow. If you then expand those Pattern nodes to show their included tools, you can see which parts of the session graph correspond to each pattern (the tools that are part of the pattern will be a subset of the session's tools, and their interconnections will match the pattern's structure). This visual mapping helps validate that the mined patterns do represent actual recurring sub-workflows.
- **Pattern Co-occurrence Network**: The CO_OCCURS_WITH relationships between Pattern nodes create a higher-level network of patterns. In this meta-graph, nodes represent entire patterns (motifs), and an edge between two pattern nodes means those patterns have been seen together in at least one session. Visualizing this can reveal clusters of patterns that commonly co-appear. You might interpret such clusters as parts of larger pipelines or related analysis steps. For example, Pattern A (quality control tools) might often co-occur with Pattern B (alignment and variant calling tools) in genomics workflows, indicating a common two-phase structure in many analyses. By adjusting the visualization (e.g., sizing nodes by their frequency or coloring by pattern size), you can get insights into which patterns are most central or which groupings of patterns frequently occur.

Overall, the Neo4j visualization allows you to **interactively explore**: - Which tools constitute each frequent pattern. - How those patterns manifest inside actual workflow instances. - Relationships between different patterns (common tools or common sessions). - The overall structure of Galaxy usage, from individual tool interactions up to recurring multi-tool motifs.

Using graph exploration features (Bloom or Browser search), you can click on a Pattern node to highlight its connections, double-click on a Session to see all its patterns and tools, etc., thereby piecing together a comprehensive picture of Galaxy workflow behaviors.

## Directory Structure

The repository is organized as follows to separate code, data, and results:

- **scripts/** - Python scripts for running the mining pipeline and related utilities:
- run_miner.py: Main entry point to execute the subgraph mining decoder with appropriate parameters. Handles environment setup (patches, multiprocessing) and constructs the argument list for the mining module. Run this to start mining patterns on the dataset.
- label_pickle.py: Utility to post-process a raw NetworkX graph pickle by adding missing attributes. It ensures the graph is directed and every edge has a numeric weight and edge_weight (required by the GNN), and assigns an id/label to nodes for readability.
- export_aura_to_pyg.py: Script demonstrating how to fetch the tool graph from a Neo4j Aura DB. It queries all Tool nodes and TOOL_CO_OCCURRENCE edges, then builds a PyTorch Geometric Data object and a NetworkX graph. It writes out files like aura_tool_graph.edgelist and aura_tool_graph.pkl. (This is useful if you need to reconstruct or update the dataset from the database).
- export_patterns_for_aura.py: After mining, use this to prepare the output for Neo4j. It reads the mined patterns pickle and converts it into a Cypher parameter format (patterns_param.cypher.txt). This makes it easy to import all patterns into Neo4j by simply setting the parameter and unwinding it in a query.
- _(Additional scripts like smoke_env.py for environment testing, and possibly results_checker.py for inspecting outputs, are provided for convenience.)_
- **data/** - Data and query files:
- aura_tool_graph.edgelist: The edgelist of the Galaxy tool usage graph (tab-separated: ToolA ToolB weight). Each line represents a directed edge from tool A to tool B with an integer weight (frequency count). This was derived from Galaxy workflow histories, where weight indicates how many times that transition occurred in the dataset.
- aura_tool_graph_attr_weighted.pkl: The NetworkX graph used as input to the miner (as discussed above). It contains the same nodes and edges as the edgelist, but in graph form with attributes. (Node attributes include id/label for the tool name; edge attributes include weight and edge_weight for frequency).
- load_tools.cypher: A Cypher script listing all unique tools as MERGE statements. Used to create the Tool nodes in Neo4j.
- patterns_param.cypher.txt: The Cypher parameter file for patterns output. Contains a single Cypher command to define a parameter \$patterns as a list of maps. Each map has keys: pid, size, edge_count, nodes (list of tool IDs in the pattern), and edges (list of {u,v} pairs representing directed edges in the pattern). This file is generated by export_patterns_for_aura.py and can be copy-pasted into Neo4j.
- (Other files like aura_tool_index.json, etc., may also be present if generated by export scripts, containing mappings of tool names to indices, but they are not directly used in the mining pipeline.)
- **results/** - Outputs generated by the mining process:
- mined_patterns_fast.pkl (or mined_patterns.pkl for full run): Pickle file containing the list of mined pattern subgraphs. Each pattern is a NetworkX graph object (with nodes labeled as tools and edges representing connections). This is the primary output of the pattern mining step.
- (If additional analysis is done, this folder could also contain things like counts.json or other analysis results, but by default it just holds the patterns.)
- **neural-subgraph-matcher-miner/** - The integrated **Neural Subgraph Learning** library which powers the subgraph mining:
- This directory is essentially a copy (or fork) of the code from the SNAP research team's _Neural Subgraph Learning_ project, which includes **NeuroMatch** (for subgraph matching) and **SPMiner** (for subgraph pattern mining). It contains sub-packages like subgraph_matching and subgraph_mining with the model definitions, training and decoder logic, etc.
- Notable subdirectories: common/ (shared model components and data utilities), subgraph_matching/ (encoder training code for the matching task), subgraph_mining/ (the decoder used for pattern mining), analyze/ (scripts for analyzing embeddings or pattern counts). We leverage the subgraph_mining.decoder module from here via our run_miner.py.
- ckpt/ inside this directory holds the pre-trained model checkpoint (model.pt). This model was trained on synthetic graphs for the subgraph matching task and is required for the decoder to work (as it provides the learned embedding space used to guide the pattern search).
- The presence of this directory in the repo means you do **not** have to separately install the library - it's included. We credit the original authors (see below) for this component. If needed, you can refer to neural-subgraph-matcher-miner/README.md for more details on the underlying algorithms and configuration options.

_(Other standard files may be present, such as a license, README (this file), etc., which are not listed above.)_

## Credits & References

This project stands on the shoulders of previous work in both the Galaxy community and graph mining research:

- **Galaxy Project** - The workflow data analyzed here comes from the Galaxy platform (<https://galaxyproject.org>). Galaxy is a widely-used scientific workflow system that enables users to create and share data analysis pipelines. We used Galaxy usage logs (tool sequence data from Galaxy workflows) as the input graph for mining. Galaxy is an open-source, web-based platform for data-intensive biomedical research and beyond[\[1\]](https://en.wikipedia.org/wiki/Galaxy_%28computational_biology%29#:~:text=Galaxy,throughput%5B%2012). We thank the Galaxy community for making such workflow data available, as it enables interesting analyses like this mining of common patterns.
- **Neural Subgraph Matching & SPMiner** - The core algorithm used for pattern mining is adapted from the **Neural Subgraph Learning** library by researchers at Stanford (Snap group). In particular, the **Subgraph Pattern Miner (SPMiner)** approach proposed by Ying et al. (2023/2024) is employed for finding frequent subgraphs. SPMiner is a GNN-based framework to extract frequent subgraph patterns from a graph dataset[\[2\]](https://github.com/snap-stanford/neural-subgraph-learning-GNN#:~:text=Frequent%20Subgraph%20Mining). This project includes code from the open-source repository that implements NeuroMatch and SPMiner. Credit goes to the original authors: Rex Ying, Tianyu Fu, Andrew Wang, Jiaxuan You, Yu Wang, and Jure Leskovec for their work on representation learning for subgraph mining. The pre-trained model (model.pt) provided here comes from their released resources, and the mining process closely follows their methodology (Greedy order-embedding guided search for motifs). For more details, see the paper _"Representation Learning for Frequent Subgraph Mining"_ (Ying et al., 2024) and the Neural Subgraph Learning GitHub repository.
- **Inspiration and Related Work** - We also acknowledge that the idea of mining frequent patterns in scientific workflows has been explored in prior works (e.g., there are studies on discovering common sub-workflows in bioinformatics pipelines). This project brings a modern GNN-based approach to that problem. Additionally, the integration with Neo4j was inspired by the need to intuitively explore graph patterns - combining data mining with graph databases for analysis.

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this code under the terms of MIT. (See the LICENSE file for details, if provided, or the text of the MIT license.)

[\[1\]](https://en.wikipedia.org/wiki/Galaxy_%28computational_biology%29#:~:text=Galaxy,throughput%5B%2012) Galaxy (computational biology) - Wikipedia

[https://en.wikipedia.org/wiki/Galaxy_(computational_biology)](https://en.wikipedia.org/wiki/Galaxy_%28computational_biology%29)

[\[2\]](https://github.com/snap-stanford/neural-subgraph-learning-GNN#:~:text=Frequent%20Subgraph%20Mining) GitHub - snap-stanford/neural-subgraph-learning-GNN

<https://github.com/snap-stanford/neural-subgraph-learning-GNN>