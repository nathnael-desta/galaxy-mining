import argparse

def parse_decoder(parser):
    dec_parser = parser.add_argument_group()
    
    # Sampling parameters
    dec_parser.add_argument('--chunk_size', type=int, default=10000,
                        help='Chunk size for processing large graphs')
    dec_parser.add_argument('--sample_method', type=str,
        help='"tree" or "radial" sampling method')
    dec_parser.add_argument('--radius', type=int,
        help='radius of node neighborhoods')
    dec_parser.add_argument('--subgraph_sample_size', type=int,
        help='number of nodes to take from each neighborhood')
    dec_parser.add_argument('--use_whole_graphs', action="store_true",
        help="whether to cluster whole graphs or sampled node neighborhoods")
        
    # Pattern search parameters
    dec_parser.add_argument('--min_pattern_size', type=int,
        help='minimum size of patterns to find')
    dec_parser.add_argument('--max_pattern_size', type=int,
        help='maximum size of patterns to find')
    dec_parser.add_argument('--min_neighborhood_size', type=int,
        help='minimum neighborhood size to consider')
    dec_parser.add_argument('--max_neighborhood_size', type=int,
        help='maximum neighborhood size to consider')
    dec_parser.add_argument('--n_neighborhoods', type=int,
        help='number of neighborhoods to sample')
    
    # Search strategy parameters
    dec_parser.add_argument('--search_strategy', type=str,
        help='"greedy" or "mcts" search strategy')
    dec_parser.add_argument('--n_trials', type=int,
        help='number of search trials to run')
    dec_parser.add_argument('--out_batch_size', type=int,
        help='number of motifs to output per graph size')
    
    # Memory efficiency parameters
    dec_parser.add_argument('--memory_efficient', action='store_true',
        help='Use memory efficient search for large graphs')
    # Beam search parameter
    parser.add_argument('--beam_width', type=int, default=5,
                        help='Width of beam for beam search')
    # Output and analysis
    dec_parser.add_argument('--out_path', type=str,
        help='path to output candidate motifs')
    dec_parser.add_argument('--analyze', action="store_true",
        help='enable analysis mode')
    dec_parser.add_argument('--motif_dataset', type=str,
        help='Motif dataset to use')
    dec_parser.add_argument('--n_clusters', type=int,
        help='number of clusters for analysis')

    # Graph type selection
    dec_parser.add_argument('--graph_type', type=str,
        help='"directed" or "undirected" graph type')
    
    # Set default values
    parser.set_defaults(
        # Dataset defaults
        dataset="enzymes",
        batch_size=1000,
        
        # Decoder defaults
        out_path="results/out-patterns.p",
        n_neighborhoods=2000,
        n_trials=1000,
        decode_thresh=0.5,
        radius=3,
        subgraph_sample_size=0,
        sample_method="tree",
        skip="learnable",
        graph_type="undirected",
        min_pattern_size=5,
        max_pattern_size=10,
        min_neighborhood_size=5,
        max_neighborhood_size=10,
        search_strategy="greedy",
        out_batch_size=10,
        node_anchored=True,
        memory_limit=1000000
    )