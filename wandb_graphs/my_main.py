from Wandb_Plotting_Tool.plot import Metric, Run, TableInfo, plot_metrics_and_runs


### Plots 1 -- all three base models
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Linear Attention and Softmax Attention Loss (Train)",
            metric_graph_out_dir="figures/1_base_models",
            metric_graph_filename="1_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Linear Attention and Softmax Attention Loss (Test)",
            metric_graph_out_dir="figures/1_base_models",
            metric_graph_filename="1_test_loss.svg",
        ),
    ],
    runs=[
        Run(
            run_name="zfdskd9t",
            run_name_plot="Linear",
            run_color="red"
        ),
        Run(
            run_name="qbcs9fqx",
            run_name_plot="Mamba",
            run_color="green"
        ),
        Run(
            run_name="ea9ph22z",
            run_name_plot="Softmax",
            run_color="blue"
        ),
    ],
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=False,
        num_decimals=2
    ),
)


### Plots 2 -- Linear attention sm norm vs out norm
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Linear Attention Norm Types Loss (Train)",
            metric_graph_out_dir="figures/2_lin_attn_norm_type",
            metric_graph_filename="2_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Linear Attention Norm Types Loss (Test)",
            metric_graph_out_dir="figures/2_lin_attn_norm_type",
            metric_graph_filename="2_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="zfdskd9t",
            run_name_plot="SM Norm + ReLU",
            run_color="red"
        ),
        Run(
            run_name="34nsedgl",
            run_name_plot="Out Norm",
            run_color="green"
        ),
        Run(
            run_name="opgr66ja",
            run_name_plot="Out Norm + ReLU",
            run_color="blue"
        ),
    ],
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)

### Plots 3 -- Convolution sizes
import math
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Various Convolution Types Loss (Train)",
            metric_graph_out_dir="figures/3_conv_sizes",
            metric_graph_filename="3_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Various Convolution Types Loss (Test)",
            metric_graph_out_dir="figures/3_conv_sizes",
            metric_graph_filename="3_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="opgr66ja",
            run_name_plot="ReLU",
            run_color="black"
        ),
        Run(
            run_name="mdcyn7ht",
            run_name_plot="Conv (w=2)",
            run_color="pink"
        ),
        Run(
            run_name="gd9f0wc9",
            run_name_plot="Conv (w=3)",
            run_color="orange"
        ),
        Run(
            run_name="s41u75rw",
            run_name_plot="Conv (w=4)",
            run_color="red"
        ),
        Run(
            run_name="226lje8f",
            run_name_plot="Conv (w=2) + SiLU",
            run_color="light blue"
        ),
        Run(
            run_name="cyw6c02o",
            run_name_plot="Conv (w=3) + SiLU",
            run_color="blue"
        ),
        Run(
            run_name="m1zp0qur",
            run_name_plot="Conv (w=4) + SiLU",
            run_color="purple"
        ),
    ],
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)

### Plots 4 -- Isolated component tests 
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Isolated Component Losses (Train)",
            metric_graph_out_dir="figures/4_isolated",
            metric_graph_filename="4_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Isolated Component Losses (Test)",
            metric_graph_out_dir="figures/4_isolated",
            metric_graph_filename="4_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="opgr66ja",
            run_name_plot="ReLU",
            run_color="black"
        ),
        Run(
            run_name="mdcyn7ht",
            run_name_plot="Conv (w=2)",
            run_color="purple"
        ),
        Run(
            run_name="dpgw7quz",
            run_name_plot="D res",
            run_color="orange"
        ),
        Run(
            run_name="jt66mjod",
            run_name_plot="Z gate",
            run_color="blue"
        ),
        Run(
            run_name="iiwxk216",
            run_name_plot="discretize",
            run_color="green"
        ),
        Run(
            run_name="vittt6s5",
            run_name_plot="A mask original",
            run_color="gray"
        ),
        Run(
            run_name="fgrw5gcn",
            run_name_plot="A mask softplus",
            run_color="red"
        ),
    ],
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)


### For table of all isolated tests
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Train loss For Linear Attention Norm Types",
            metric_graph_out_dir="figures/T1_isolated_comp_table",
            metric_graph_filename="T1_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Test Loss For Linear Attention Norm Types",
            metric_graph_out_dir="figures/T1_isolated_comp_table",
            metric_graph_filename="T1_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="zfdskd9t",
            run_name_plot="SM Norm + ReLU",
            run_color="red"
        ),
        Run(
            run_name="34nsedgl",
            run_name_plot="Out Norm",
            run_color="green"
        ),
        Run(
            run_name="opgr66ja",
            run_name_plot="Out Norm + ReLU",
            run_color="blue"
        ),
        Run(
            run_name="mdcyn7ht",
            run_name_plot="Out Norm + Conv (w=2)",
            run_color="red"
        ),
        Run(
            run_name="gd9f0wc9",
            run_name_plot="Out Norm + Conv (w=3)",
            run_color="green"
        ),
        Run(
            run_name="226lje8f",
            run_name_plot="Out Norm + Conv (w=2) + SiLU",
            run_color="blue"
        ),
        Run(
            run_name="dpgw7quz",
            run_name_plot="Out Norm + D res",
            run_color="orange"
        ),
        Run(
            run_name="jt66mjod",
            run_name_plot="Out Norm + Z gate",
            run_color="blue"
        ),
        Run(
            run_name="iiwxk216",
            run_name_plot="Out Norm + discretize",
            run_color="green"
        ),
        Run(
            run_name="vittt6s5",
            run_name_plot="Out Norm + A mask original",
            run_color="green"
        ),
        Run(
            run_name="fgrw5gcn",
            run_name_plot="Out Norm + A mask softplus",
            run_color="red"
        ),
    ],
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)

### Plots 5 -- Mamba buildup main
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Algorithm Buildup (Train)",
            metric_graph_out_dir="figures/5_buildup_main",
            metric_graph_filename="5_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Algorithm Buildup (Test)",
            metric_graph_out_dir="figures/5_buildup_main",
            metric_graph_filename="5_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="34nsedgl",
            run_name_plot="Plain Linear",
            run_color="black"
        ),
        Run(
            run_name="mdcyn7ht",
            run_name_plot="Conv (w=2)",
            run_color="red"
        ),
        Run(
            run_name="7klpx6ml",
            run_name_plot="Conv (w=2) + A mask original",
            run_color="green"
        ),
        Run(
            run_name="2n0t80gy",
            run_name_plot="Conv (w=2) + A mask softplus",
            run_color="blue"
        ),
        Run(
            run_name="na9afliq",
            run_name_plot="Conv (w=2) + A mask softplus + discretize",
            run_color="purple"
        ),
    ],
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)



### Plots 6 -- Mamba buildup redundant
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Algorithm Buildup - Redundant Additions (Train)",
            metric_graph_out_dir="figures/6_buildup_red",
            metric_graph_filename="6_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Algorithm Buildup - Redundant Additions (Test)",
            metric_graph_out_dir="figures/6_buildup_red",
            metric_graph_filename="6_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="na9afliq",
            run_name_plot="Conv (w=2) + A mask softplus + discretize",
            run_color="black"
        ),
        Run(
            run_name="35n4y5h0",
            run_name_plot="Conv (w=2) + A mask softplus + discretize + SiLU",
            run_color="red"
        ),
        Run(
            run_name="6sihihh2",
            run_name_plot="Conv (w=2) + A mask softplus + discretize + D res",
            run_color="green"
        ),
        Run(
            run_name="a3echla3",
            run_name_plot="Conv (w=2) + A mask softplus + discretize + Z gate",
            run_color="blue"
        ),
    ],
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)



### For table of all mamba buildups
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Algorithm Buildup (Train)",
            metric_graph_out_dir="figures/T2_mamba_buildup_table",
            metric_graph_filename="T2_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Algorithm Buildup (Test)",
            metric_graph_out_dir="figures/T2_mamba_buildup_table",
            metric_graph_filename="T2_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="34nsedgl",
            run_name_plot="Plain Linear",
            run_color="black"
        ),
        Run(
            run_name="mdcyn7ht",
            run_name_plot="Conv (w=2)",
            run_color="red"
        ),
        Run(
            run_name="7klpx6ml",
            run_name_plot="Conv (w=2) + A mask original",
            run_color="yellow"
        ),
        Run(
            run_name="2n0t80gy",
            run_name_plot="Conv (w=2) + A mask softplus",
            run_color="green"
        ),
        Run(
            run_name="na9afliq",
            run_name_plot="Conv (w=2) + A mask softplus + discretize",
            run_color="blue"
        ),
        Run(
            run_name="35n4y5h0",
            run_name_plot="Conv (w=2) + A mask softplus + discretize + SiLU",
            run_color="purple"
        ),
        Run(
            run_name="6sihihh2",
            run_name_plot="Conv (w=2) + A mask softplus + discretize + D res",
            run_color="pink"
        ),
        Run(
            run_name="a3echla3",
            run_name_plot="Conv (w=2) + A mask softplus + discretize + Z gate",
            run_color="grey"
        ),
    ],
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)



### Graph 7 - Current method is still worse than softmax
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Algorithm Variants and Softmax Loss (Train)",
            metric_graph_out_dir="figures/7_sm_comp",
            metric_graph_filename="7_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Algorithm Variants and Softmax Loss (Test)",
            metric_graph_out_dir="figures/7_sm_comp",
            metric_graph_filename="7_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="na9afliq",
            run_name_plot="Conv (w=2) + A mask softplus + discretize",
            run_color="red"
        ),
        Run(
            run_name="yjdb08ea",
            run_name_plot="Squared + Conv (w=2) + A mask softplus + discretize",
            run_color="blue"
        ),
        Run(
            run_name="qbcs9fqx",
            run_name_plot="Mamba",
            run_color="green"
        ),
        Run(
            run_name="ea9ph22z",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)





### Graph 8.1 - Scaling the current method (small, 2048)
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Small Model - 2048 Sequence Length (Train)",
            metric_graph_out_dir="figures/8.1_full_comp",
            metric_graph_filename="8.1_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Small Model - 2048 Sequence Length (Test)",
            metric_graph_out_dir="figures/8.1_full_comp",
            metric_graph_filename="8.1_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="yjdb08ea",
            run_name_plot="Proposed",
            run_color="red"
        ),
        Run(
            run_name="qbcs9fqx",
            run_name_plot="Mamba",
            run_color="green"
        ),
        Run(
            run_name="ea9ph22z",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)


### Graph 8.2 - Scaling the current method (small, 4096)
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Small Model - 4096 Sequence Length (Train)",
            metric_graph_out_dir="figures/8.2_full_comp",
            metric_graph_filename="8.2_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Small Model - 4096 Sequence Length (Test)",
            metric_graph_out_dir="figures/8.2_full_comp",
            metric_graph_filename="8.2_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="do62gedv",
            run_name_plot="Proposed",
            run_color="red"
        ),
        Run(
            run_name="uo1nhef4",
            run_name_plot="Mamba",
            run_color="green"
        ),
        Run(
            run_name="eevxka2s",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)


### Graph 8.3 - Scaling the current method (small, 8192)
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Small Model - 8192 Sequence Length (Train)",
            metric_graph_out_dir="figures/8.3_full_comp",
            metric_graph_filename="8.3_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Small Model - 8192 Sequence Length (Test)",
            metric_graph_out_dir="figures/8.3_full_comp",
            metric_graph_filename="8.3_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="gwt9v002",
            run_name_plot="Proposed",
            run_color="red"
        ),
        Run(
            run_name="r06dnukl",
            run_name_plot="Mamba",
            run_color="green"
        ),
        Run(
            run_name="3p9mf2r3",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)


### Graph 8.4 - Scaling the current method (medium, 2048)
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Medium Model - 2048 Sequence Length (Train)",
            metric_graph_out_dir="figures/8.4_full_comp",
            metric_graph_filename="8.4_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Medium Model - 2048 Sequence Length (Test)",
            metric_graph_out_dir="figures/8.4_full_comp",
            metric_graph_filename="8.4_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="b85jtwri",
            run_name_plot="Proposed",
            run_color="red"
        ),
        Run(
            run_name="gmbstg9n",
            run_name_plot="Mamba",
            run_color="green"
        ),
        Run(
            run_name="e048zpqr",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)


### Graph 8.5 - Scaling the current method (medium, 4096)
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Medium Model - 4096 Sequence Length (Train)",
            metric_graph_out_dir="figures/8.5_full_comp",
            metric_graph_filename="8.5_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Medium Model - 4096 Sequence Length (Test)",
            metric_graph_out_dir="figures/8.5_full_comp",
            metric_graph_filename="8.5_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="zsi4ijie",
            run_name_plot="Proposed",
            run_color="red"
        ),
        Run(
            run_name="z1rbud4l",
            run_name_plot="Mamba",
            run_color="green"
        ),
        Run(
            run_name="ktic0mne",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)


### Graph 8.6 - Scaling the current method (medium, 8192)
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Medium Model - 8192 Sequence Length (Train)",
            metric_graph_out_dir="figures/8.6_full_comp",
            metric_graph_filename="8.6_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Medium Model - 8192 Sequence Length (Test)",
            metric_graph_out_dir="figures/8.6_full_comp",
            metric_graph_filename="8.6_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="uaaza0es",
            run_name_plot="Proposed",
            run_color="red"
        ),
        Run(
            run_name="ib3m3kgq",
            run_name_plot="Mamba",
            run_color="green"
        ),
        Run(
            run_name="f18tgujl",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)

### Graph 9 - Unstable training dynamics
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Discretization Variant Instability Losses (Train)",
            metric_graph_out_dir="figures/9_unstable",
            metric_graph_filename="9_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Discretization Variant Instability Losses (Test)",
            metric_graph_out_dir="figures/9_unstable",
            metric_graph_filename="9_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="pnacknd1",
            run_name_plot="No Discretization",
            run_color="blue"
        ),
        Run(
            run_name="tla78qdo",
            run_name_plot="With Discretization",
            run_color="red"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)

### Graph 10 - Exp Variation (2048 seq len)
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Medium Model - 2048 Sequence Length (Train)",
            metric_graph_out_dir="figures/10.1_exp",
            metric_graph_filename="10.1_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Medium Model - 2048 Sequence Length (Test)",
            metric_graph_out_dir="figures/10.1_exp",
            metric_graph_filename="10.1_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="gmbstg9n",
            run_name_plot="Mamba",
            run_color="blue"
        ),
        Run(
            run_name="b85jtwri",
            run_name_plot="Squared",
            run_color="green"
        ),
        Run(
            run_name="zx6wcpn2",
            run_name_plot="Exp",
            run_color="red"
        ),
        Run(
            run_name="e048zpqr",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)
### Graph 10 - Exp Variation (4096 seq len)
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Medium Model - 4096 Sequence Length (Train)",
            metric_graph_out_dir="figures/10.2_exp",
            metric_graph_filename="10.2_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Medium Model - 4096 Sequence Length (Test)",
            metric_graph_out_dir="figures/10.2_exp",
            metric_graph_filename="10.2_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="z1rbud4l",
            run_name_plot="Mamba",
            run_color="blue"
        ),
        Run(
            run_name="zsi4ijie",
            run_name_plot="Squared",
            run_color="green"
        ),
        Run(
            run_name="su8ppg08",
            run_name_plot="Exp",
            run_color="red"
        ),
        Run(
            run_name="ktic0mne",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)
### Graph 10 - Exp Variation (8192 seq len)
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="Medium Model - 8192 Sequence Length (Train)",
            metric_graph_out_dir="figures/10.3_exp",
            metric_graph_filename="10.3_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="Medium Model - 8192 Sequence Length (Test)",
            metric_graph_out_dir="figures/10.3_exp",
            metric_graph_filename="10.3_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="ib3m3kgq",
            run_name_plot="Mamba",
            run_color="blue"
        ),
        Run(
            run_name="uaaza0es",
            run_name_plot="Squared",
            run_color="green"
        ),
        Run(
            run_name="pnacknd1",
            run_name_plot="Exp",
            run_color="red"
        ),
        Run(
            run_name="f18tgujl",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)
"""

"""
### Graph 11 - Long run + NIAH
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="NIAH Long Training Run (Train)",
            metric_graph_out_dir="figures/11_niah",
            metric_graph_filename="11_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="NIAH Long Training Run (Test)",
            metric_graph_out_dir="figures/11_niah",
            metric_graph_filename="11_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="3b344ay9",
            run_name_plot="Mamba",
            run_color="blue"
        ),
        Run(
            run_name="d8sh2h6s",
            run_name_plot="Squared",
            run_color="green"
        ),
        Run(
            run_name="txhu1vhg",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=400_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)


### Graph 12 - The Pile
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="The Pile Train Loss",
            metric_graph_out_dir="figures/12_pile",
            metric_graph_filename="12_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="The Pile Test Loss",
            metric_graph_out_dir="figures/12_pile",
            metric_graph_filename="12_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="43kdcjk5",
            run_name_plot="Mamba",
            run_color="blue"
        ),
        Run(
            run_name="bz7ovl6u",
            run_name_plot="Squared",
            run_color="green"
        ),
        Run(
            run_name="ky32ktw9",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=400_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)


### Graph 13 - SlimPajama
plot_metrics_and_runs(
    metrics=[
        Metric(
            metric_name="loss",
            metric_name_plot="Train loss",
            metric_graph_title="SlimPajama Train Loss",
            metric_graph_out_dir="figures/13_slimpj",
            metric_graph_filename="13_train_loss.svg",
            metric_n_step_avg=1000
        ),
        Metric(
            metric_name="test_loss",
            metric_name_plot="Test loss",
            metric_graph_title="SlimPajama Test Loss",
            metric_graph_out_dir="figures/13_slimpj",
            metric_graph_filename="13_test_loss.svg",
        ),
    ],
    
    
    
    runs=[
        Run(
            run_name="np9vmjxl",
            run_name_plot="Mamba",
            run_color="blue"
        ),
        Run(
            run_name="9bjzvg18",
            run_name_plot="Squared",
            run_color="green"
        ),
        Run(
            run_name="wdagmenq",
            run_name_plot="Softmax",
            run_color="black"
        ),
    ],

    
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=400_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)