PipeDream

PipeDream: Generalized Pipeline Parallelism for DNN Training

ABSTRACT

DNNtraining is extremely time-consuming, necessitating efficient
 multi-accelerator parallelization. Current approaches to paralleliz
ing training primarily use intra-batch parallelization, where a single
 iteration of training is split over the available workers, but suffer
 from diminishing returns at higher worker counts. We present
 PipeDream, a system that adds inter-batch pipelining to intra-batch
 parallelism to further improve parallel training throughput, help
ing to better overlap computation with communication and reduce
 the amount of communication when possible. Unlike traditional
 pipelining, DNN training is bi-directional, where a forward pass
 through the computation graph is followed by a backward pass that
 uses state and intermediate data computed during the forward pass.
 Naïve pipelining can thus result in mismatches in state versions
 used in the forward and backward passes, or excessive pipeline
 f
 lushes and lower hardware efficiency. To address these challenges,
 PipeDream versions model parameters for numerically correct gra
dient computations, and schedules forward and backward passes
 of different minibatches concurrently on different workers with
 minimal pipeline stalls. PipeDream also automatically partitions
 DNNlayers among workers to balance work and minimize com
munication. Extensive experimentation with a range of DNN tasks,
 models, and hardware configurations shows that PipeDream trains
 models to high accuracy up to 5.3× faster than commonly used
 intra-batch parallelism techniques.
 深度神经网络训练极其耗时，因此需要高效的多加速器并行化方案。当前主流的训练并行方法主要采用批内并行技术——将单次训练迭代拆分到多个工作节点执行，但随着工作节点数量增加，其性能提升会呈现收益递减现象。我们提出PipeDream系统，通过在批内并行基础上引入批间流水线技术，进一步提升了并行训练吞吐量：该系统能更好地实现计算与通信的重叠，并在可能情况下减少通信量。与传统流水线不同，DNN训练包含双向计算流程，在前向传播完成计算图遍历后，紧接着需要利用前向传播产生的状态和中间数据进行反向传播。若直接采用朴素流水线方案，会导致前向与反向传播使用的状态版本不匹配，或引发频繁的流水线清空及硬件效率下降。为解决这些挑战，PipeDream通过版本化模型参数来确保梯度计算的数值正确性，同时采用最小化流水线停滞的调度策略，使不同微批次的前向传播与反向传播能够在不同工作节点上并发执行。此外，PipeDream还能自动在工作节点间划分DNN网络层，以实现负载均衡并最小化通信开销。通过在多种DNN任务、模型架构和硬件配置上的广泛实验表明，PipeDream能以最高5.3倍的速度提升，将模型训练达到目标精度，显著优于当前常用的批内并行技术。
 
 1.INTRODUCTION
 
   DeepNeuralNetworks(DNNs)havefacilitatedtremendousprogress
 across a range of applications, including image classification [26,
 37, 48], translation [55], language modeling [40], and video caption
ing [54]. As DNNs have become more widely deployed, they have
 also become more computationally expensive to train, thus requir
ing parallel execution across multiple accelerators (e.g., GPUs).
深度神经网络（DNNs）的广泛应用推动了一系列领域的重大进展，涵盖图像分类[26, 37, 48]、机器翻译[55]、语言建模[40]及视频内容描述[54]等领域。随着深度神经网络部署范围的不断扩大，其训练过程对计算资源的需求也日益增长，因此需要跨多个加速器（如图形处理器GPU）进行并行计算处理。
  
   DNNtraining proceeds in iterations of forward and backward
 pass computations. In each iteration, the training loop processes a
 minibatch of input data and performs an update to the model pa
rameters. Current approaches focus on parallelizing each iteration
 of the optimization algorithm across a set of workers. For exam
ple, data parallelism partitions the input data across workers [37], model parallelism partitions operators across workers [16, 21], and
 hybrid schemes partition both [33, 34, 36]. Unfortunately, intra
batch parallelization can suffer from high communication costs at
 large scale. For example, Figure 1 shows the communication over
head for data parallelism across five different DNN models on three
 different types of multi-GPU servers. Over 32 GPUs, the communi
cation overhead for some models, computed as the percentage of
 total time spent on communication stalls, is as high as 90% due to
 expensive cross-server all_reduce communication. Communica
tion overheads are high even on servers where GPUs within the
 server are connected by dedicated interconnects like NVLink [4].
 Moreover, rapid increases in GPU compute capacity over time will
 further shift the bottleneck of training towards communication for
 all models.
深度神经网络的训练以前向传播和反向传播的迭代计算方式进行。在每次迭代中，训练流程会处理一个微型批次的输入数据，并对模型参数执行更新操作。当前主流方法主要致力于在多个工作节点间并行执行优化算法的每次迭代：数据并行方案将输入数据划分到不同工作节点[37]，模型并行方案将运算操作符分配到不同工作节点[16,21]，而混合方案则对数据和模型同时进行划分[33,34,36]。然而，这种批内并行方案在大规模部署时会面临高昂的通信开销。如图1所示，在三种不同类型的多GPU服务器上对五种DNN模型进行数据并行训练时的通信开销表明，当使用32个GPU时，由于昂贵的跨服务器全局归约通信，某些模型的通信开销（以通信等待时间占总时长的百分比计算）高达90%。即使在配备NVLink等专用互联技术的服务器内部，GPU间的通信开销依然可观。随着GPU计算能力的快速提升，通信瓶颈在未来将成为所有模型训练过程中日益突出的制约因素。


<img width="847" height="256" alt="image" src="https://github.com/user-attachments/assets/0191a0b8-81f4-4ea7-af74-c839ebe12b21" />

 
   In this paper, we propose PipeDream, a system that uses pipeline
 parallelism to enable faster DNN training by combining intra-batch
 parallelism with inter-batch parallelization. PipeDream divides the
 model among available workers, assigning a group of consecutive
 operators (called layers in DNN terminology) in the operator graph
 to each of them, and then overlaps the computation and commu
nication of different inputs in a pipelined fashion. This process
 can greatly reduce inter-worker communication because it limits
 the communication to layer inputs and outputs (activations in the
 forward pass and gradients in the backward pass) solely across
 consecutive layers assigned to different workers, which for many
 modelsaremuchsmallerthanthesizeoftheentiremodel.Moreover,
 this communication is peer-to-peer, as opposed to all-to-all.
 While pipelining is a simple idea, DNN training poses an impor
tant challenge not present in traditional pipelining: DNN training
 is bi-directional—the forward pass is followed by a backward pass
 through the same layers in reverse order, using state and interme
diate results from the forward pass. To keep the pipeline full and
 thus achieve high hardware efficiency, a naïve scheduling mech
anism might inject all minibatches in an epoch into the pipeline,
 f
 irst completing forward passes for all input minibatches followed
 by backward passes. However, this approach suffers from low sta
tistical efficiency [18], increasing the number of passes through
 the dataset needed to produce a high-quality model. Furthermore,
 this strategy could prevent the model from reaching the desired
 target accuracy, since gradients are averaged over all training sam
ples [10, 39]. To improve statistical efficiency, one could inject only
 a subset ofm minibatches into the pipeline, and apply weight up
dates every m minibatches, as recently proposed by GPipe [28].
 However, this reduces hardware efficiency due to more frequent
 pipeline flushes. Traditional model parallel training corresponds to
 an extreme case of this (m = 1).
 本文提出PipeDream系统，该系统通过将批内并行与批间并行化相结合，采用流水线并行技术实现更高效的深度神经网络训练。PipeDream将模型在可用工作节点间进行划分，为每个节点分配计算图中一组连续的运算单元（在DNN术语中称为层），随后以流水线方式实现不同输入数据在计算与通信过程中的重叠执行。这种方法能显著降低工作节点间的通信开销，因为通信仅限于分配给不同工作节点的连续层之间的层输入输出（前向传播中的激活值和反向传播中的梯度），对于多数模型而言，其数据量远小于完整模型参数规模。此外，这种通信采用点对点模式，而非全互联通信。
尽管流水线是简明直观的概念，但DNN训练却带来了传统流水线中不存在的重要挑战：DNN训练具有双向特性——前向传播结束后需要沿相同层结构逆序执行反向传播，且需使用前向传播产生的状态和中间结果。为保持流水线充盈从而实现硬件高效性，简单调度机制可能会将整个训练周期中的所有微批次数据注入流水线，先完成所有输入微批次的前向传播再执行反向传播。然而这种方法会降低统计效率[18]，增加生成高质量模型所需遍历数据集的次数。更严重的是，由于梯度需在所有训练样本上进行平均[10,39]，该策略可能阻碍模型达到预期目标精度。
为提高统计效率，可借鉴GPipe[28]近期提出的方案，仅将m个微批次数据子集注入流水线，并每处理m个微批次后更新权重参数。但这种方法会因更频繁的流水线清空而降低硬件效率。传统模型并行训练正是这种方案的极端案例（m=1）。

 PipeDream takes a more nuanced approach to pipelining that
 outperforms other solutions– it achieves high hardware efficiency
 with no pipeline stalls in steady state, and high statistical efficiency
  comparable to data parallelism using the same number of workers.
 Given a pipeline of groups of consecutive layers executed on differ
ent workers (called a stage), PipeDream uses a scheduling algorithm
 called 1F1B to keep hardware well utilized while achieving seman
tics similar to data parallelism. In 1F1B’s steady state, each worker
 strictly alternates between forward and backward passes for its
 stage, ensuring high resource utilization (negligible pipeline stalls,
 no pipeline flushes) even in the common case where the backward
 pass takes longer than the forward pass. 1F1B also uses different
 versions of model weights to maintain statistical efficiency com
parable to data parallelism. Each backward pass in a stage results
 in weight updates; the next forward pass uses the latest version
 of weights available, and “stashes" a copy of these weights to use
 during the corresponding backward pass. Although the forward
 pass will not see updates from incomplete in-flight mini-batches,
 learning is still effective because model weights change relatively
 slowly and bounded staleness has been found effective in improv
ing training speeds [19, 43]. However, for the backward pass to
 compute numerically correct gradients, the same weight version
 used during the forward pass must be used. PipeDream limits the
 number of “in-pipeline” minibatches to the minimum needed to
 keep the pipeline full, reducing memory overhead.
PipeDream采用了一种更为精细的流水线处理方案，其性能优于现有解决方案——该系统在稳定运行状态下既能实现无流水线停滞的高硬件效率，又能在使用相同数量工作节点的条件下达到与数据并行相媲美的统计效率。
通过构建由多个工作节点分别执行连续层组（称为阶段）的流水线架构，PipeDream采用名为1F1B的调度算法，在保持与数据并行相似语义的同时实现硬件资源的高效利用。在1F1B的稳定运行状态下，每个工作节点会严格交替执行其对应阶段的前向传播与反向传播，即使在反向传播耗时通常长于前向传播的常见场景下，也能确保较高的资源利用率（实现可忽略的流水线阻塞且无需流水线清空）。
为维持与数据并行相当的统计效率，1F1B算法采用多版本权重管理机制：每个阶段完成反向传播后立即更新权重参数，随后的前向传播将使用可获取的最新权重版本，同时"暂存"这些权重的副本以供对应反向传播计算使用。尽管前向传播无法获取正在处理中的未完成微批次所产生的参数更新，但由于模型权重变化相对缓慢，且已有研究表明有限程度的延迟更新能有效提升训练速度[19,43]，该机制仍能保证训练效果。为确保梯度计算的数值准确性，反向传播必须严格使用前向传播时采用的权重版本。
通过将流水线中并行的微批次数量控制在维持流水线饱和所需的最小值，PipeDream有效降低了内存开销。

  comparable to data parallelism using the same number of workers.
 Given a pipeline of groups of consecutive layers executed on differ
ent workers (called a stage), PipeDream uses a scheduling algorithm
 called 1F1B to keep hardware well utilized while achieving seman
tics similar to data parallelism. In 1F1B’s steady state, each worker
 strictly alternates between forward and backward passes for its
 stage, ensuring high resource utilization (negligible pipeline stalls,
 no pipeline flushes) even in the common case where the backward
 pass takes longer than the forward pass. 1F1B also uses different
 versions of model weights to maintain statistical efficiency com
parable to data parallelism. Each backward pass in a stage results
 in weight updates; the next forward pass uses the latest version
 of weights available, and “stashes" a copy of these weights to use
 during the corresponding backward pass. Although the forward
 pass will not see updates from incomplete in-flight mini-batches,
 learning is still effective because model weights change relatively
 slowly and bounded staleness has been found effective in improv
ing training speeds [19, 43]. However, for the backward pass to
 compute numerically correct gradients, the same weight version
 used during the forward pass must be used. PipeDream limits the
 number of “in-pipeline” minibatches to the minimum needed to
 keep the pipeline full, reducing memory overhead.
 PipeDream采用了一种更为精细的流水线处理方法，其性能表现优于其他解决方案——该系统在稳定状态下能够实现高硬件利用率且无流水线阻塞，同时在使用相同数量工作节点的情况下，可获得与数据并行相媲美的高统计效率。通过构建在多个工作节点上执行连续层组（称为阶段）的流水线架构，PipeDream采用名为1F1B的调度算法，在保持与数据并行相似语义的同时实现硬件资源的高效利用。
在1F1B调度算法的稳定运行状态下，每个工作节点会严格交替执行其负责阶段的前向传播与反向传播，即使在反向传播耗时普遍长于前向传播的典型场景下，仍能确保高资源利用率（实现可忽略的流水线阻塞与零流水线清空）。为维持与数据并行相当的统计效率，1F1B还采用了多版本权重管理机制：每个阶段完成反向传播后立即更新权重参数，而后续前向传播则使用最新的权重版本，同时保存该版本权重的副本供对应反向传播计算使用。尽管前向传播无法实时获取正在处理中的微批次所产生的参数更新，但由于模型权重变化相对缓慢，且已有研究证明有限程度的延迟更新能有效加速训练过程[19,43]，该机制仍能保证训练效果。
为确保梯度计算的数值准确性，反向传播必须使用与前向传播完全一致的权重版本。PipeDream通过将流水线中并行的微批次数量控制在刚好维持流水线饱和的最小需求值，有效降低了内存开销。

 Operating the pipeline at peak throughput also requires that
 all stages in the pipeline take roughly the same amount of time,
 since the throughput of a pipeline is bottlenecked by the slowest
 stage. PipeDream automatically determines how to partition the
 operators of the DNN based on a short profiling run performed on
 a single GPU, balancing computational load among the different
 stages while minimizing communication for the target platform.
 PipeDream effectively load balances even in the presence of model
 diversity (computation and communication) and platform diversity
 (interconnect topologies and hierarchical bandwidths). As DNNs do
 not always divide evenly among available workers, PipeDream may
 decide to use data parallelism for some stages—multiple workers
 can be assigned to a given stage, processing different minibatches
 in parallel. Note that vanilla data parallelism corresponds to the
 pipeline having a single replicated stage. PipeDream extends 1F1B
 to incorporate round-robin scheduling across data-parallel stages,
 while making sure that gradients in a backward pass are routed to
 the corresponding worker from the forward pass since the same
 weight version and intermediate outputs need to be used for a cor
rect gradient computation. The combined scheduling algorithm,1F1B-RR, produces a static schedule of operators that each worker
 runs repeatedly, keeping utilization high across all workers. Thus,
 pipeline-parallel training can be thought of as a principled combi
nation of inter-batch pipelining with intra-batch parallelism.
为实现流水线的最佳吞吐量，必须确保管道中各阶段耗时基本均衡，因为流水线的整体吞吐量取决于最慢阶段的执行效率。PipeDream通过基于单GPU的短期性能分析，自动确定深度神经网络中运算符的最佳划分方案，在目标平台上实现各阶段计算负载均衡的同时，最小化通信开销。该系统能有效应对模型多样性（计算与通信模式差异）和平台多样性（互联拓扑与层级带宽差异）带来的负载均衡挑战。
当深度神经网络无法在可用工作节点间均匀划分时，PipeDream可针对特定阶段采用数据并行策略——为指定阶段分配多个工作节点，并行处理不同微批次数据。值得注意的是，传统数据并行方案相当于在整个流水线中仅设置单一复制阶段。PipeDream对1F1B算法进行了扩展，引入数据并行阶段间的轮询调度机制（1F1B-RR），同时确保反向传播中的梯度数据能准确路由至前向传播对应的源工作节点，这是维持梯度计算正确性所必需的（需使用相同权重版本和中间输出结果）。
这套融合后的调度算法（1F1B-RR）会生成静态的运算符执行计划，每个工作节点按计划循环执行任务，从而保持所有节点的高利用率。因此，流水线并行训练可被视为批间流水线与批内并行化方法的系统性融合。

Our evaluation, encompassing many combinations of DNN mod
els, datasets, and hardware configurations, confirms the training
 time benefits of PipeDream’s pipeline parallelism. Compared to
 data-parallel training, PipeDream reaches a high target accuracy
 on multi-GPU machines up to 5.3× faster for image classification
 tasks, up to 3.1× faster for machine translation tasks, 4.3× faster for
 language modeling tasks, and 3× faster for video captioning models.
 PipeDream is also 2.6×– 15× faster than model parallelism, up to
 1.9× faster than hybrid parallelism, and 1.7× faster than simpler
 approaches to pipelining such as GPipe’s approach.
 我们的评估涵盖了多种深度神经网络模型、数据集及硬件配置的组合，结果验证了PipeDream流水线并行技术对训练时效的提升成效。与数据并行训练相比，PipeDream在多GPU机器上达到目标精度的训练速度显著提升：图像分类任务最高提速5.3倍，机器翻译任务最高提速3.1倍，语言建模任务提速4.3倍，视频描述生成模型提速3倍。相较模型并行方案，PipeDream实现了2.6倍至15倍的加速效果；相比混合并行方案最高提升1.9倍；与GPipe等简易流水线方案相比也获得了1.7倍的性能优势。

 2 BACKGROUNDANDRELATEDWORK
 
 ADNNmodeliscomposedofmanyoperators organized into layers.
 When parallelizing DNN training, these layers may be partitioned
 over the available workers in different ways. In this section, we
 cover two broad classes of parallel DNN training: intra- and inter
batch. We also highlight the challenges posed by DNN model and
 hardware diversity for effective parallelization.
 深度神经网络模型由多个按层级结构组织的运算单元构成。在实现DNN训练并行化时，这些网络层可以通过不同方式在可用工作节点上进行划分。本节将系统阐述并行DNN训练的两大基本范式：批内并行与批间并行，并重点分析DNN模型异构性与硬件多样性为高效并行化带来的核心挑战。

  2.1 Intra-batch Parallelism
  The most common way to train DNN models today is intra-batch
 parallelization, where a single iteration of training is split across
 available workers.
 DataParallelism.Indataparallelism,inputsarepartitionedacross
 workers. Each worker maintains a local copy of the model weights
 and trains on its own partition of inputs while periodically synchro
nizing weights with other workers, using either collective commu
nication primitives like all_reduce [24] or parameter servers [38].
 The amountofdata communicated is proportional to the number of
 model weights and the number of workers participating in training.
 当前最主流的深度神经网络训练方式是批内并行，即将单次训练迭代的计算过程分配到所有可用工作节点上执行。
数据并行 是批内并行的典型实现方式：各工作节点在本地维护完整的模型参数副本，并对分配到的输入数据子集进行独立训练，同时通过集体通信原语（如全局归约all_reduce[24]）或参数服务器[38]定期进行权重同步。这种模式下，通信数据量与模型参数量及参与训练的工作节点数量成正比。

 The most commonly used form of data parallelism, referred to
 as bulk synchronous parallel or BSP [52]1, requires each worker to
 wait for gradients from other workers. Despite optimizations such
 as Wait-free Backpropagation [57], where weight gradients are sent
 as soon as they are available (common in modern frameworks),
 communication stalls are sometimes inevitable for large modelswherethetimeneededtosynchronizegradientsacrossworkers
 candominatecomputationtime.
目前最常用的数据并行形式被称为批量同步并行（Bulk Synchronous Parallel, BSP）[52]，该方案要求所有工作节点必须相互等待梯度同步完成。尽管现有优化技术（如现代深度学习框架普遍采用的"无等待反向传播"[57]）支持在梯度计算完成后立即发送，但对于大规模模型训练而言，跨节点梯度同步所需时间可能远超计算时间，导致通信阻塞问题仍难以避免。

  Figure1quantitativelyshowsthefractionoftrainingtimespent
 incommunicationstallswithdataparallelismfordifferentclassesof
 DNNsusingthreetypesofservers:8-1080TiGPUinstanceslinked
 overPCIewithinserversand25Gbpsinterconnectsacrossservers,
 4-V100GPUinstanceswithoutNVLinkand10Gbpsinterconnects
 acrossservers,and8-V100GPUinstanceswithNVLinkintercon
nectswithinserversand25Gbpsinterconnectsacrossservers.
图1定量展示了在不同类型服务器上，采用数据并行训练各类深度神经网络时通信停滞时间占训练总时长的比例。测试涵盖三种服务器配置：采用PCIe内部互联和25Gbps跨服务器互联的8路1080Ti GPU服务器；未配置NVLink且采用10Gbps跨服务器互联的4路V100 GPU服务器；以及配备NVLink内部互联和25Gbps跨服务器互联的8路V100 GPU服务器。

Wefocusonfourkeytakeaways.First,thecommunicationover
headformanyofthesemodelsishighdespiteusingmulti-GPU
 serversandstate-of-the-artcommunicationlibrarieslikeNCCL.
 DataparallelismscaleswellformodelslikeResNet-50,whichhave
 alargenumberofconvolutionallayerswithcompactweightrep
resentations,butscaleslesswellforothermodelswithLSTMor
 fully-connectedlayers,whichhavemoredenseweightrepresen
tations.Second,applicationsdistributedacrossmulti-GPUservers
 arebottleneckedbyslowerinter-serverlinks,asevidencedbycom
municationoverheadsspikingandthenplateauingwhentraining
 scalesouttomultipleservers.Dataparallelismforsuchhierarchical
 networkscanbeapoorfit, sincethesamenumberofbytesare
 sentoverbothhigh-andlow-bandwidthchannels.Third,asthe
 numberofdata-parallelworkersincreases,communicationover
headsincreaseforallmodels,eveniftrainingisperformedona
 multi-GPUinstancewithNVLink.Colemanetal.[17]showedsim
ilarresults.Fourth,asGPUcomputespeedsincrease(1080Tisto
 V100s),communicationoverheadsalsoincreaseforallmodels.
 我们主要关注四个关键发现。首先，尽管采用了多GPU服务器及NCCL等先进通信库，多数模型的通信开销依然居高不下。数据并行对ResNet-50等具有紧凑型权重表示的卷积层模型扩展性良好，但对包含LSTM或全连接层等稠密权重结构的模型扩展效果欠佳。
其次，分布式多GPU服务器应用的瓶颈主要存在于跨服务器低速链路——当训练规模扩展至多服务器时，通信开销会急剧上升并最终达到平台期。这种分层网络架构与数据并行的适配性较差，因为相同数据量需要同时通过高带宽与低带宽信道进行传输。
第三，随着数据并行节点数量增加，即使采用配备NVLink的多GPU实例进行训练，所有模型的通信开销仍会持续增长，Coleman等人[17]的研究也证实了这一现象。
第四，随着GPU计算性能升级（从1080Ti到V100），所有模型的通信开销占比均呈现上升趋势。

 OtherDPOptimizations.Asynchronousparalleltraining(ASP)
 allowseachworkertoproceedwiththenextinputminibatchbe
forereceivingthegradients fromthepreviousminibatch.This
 approachimproveshardwareefficiency(timeneededperiteration)
 overBSPbyoverlappingcomputationwithcommunication,but
 alsointroducesstalenessandreducesstatisticalefficiency(number
 ofiterationsneededtoreachaparticulartargetaccuracy)[12,20].
 其他数据并行优化方案中，异步并行训练允许工作节点在接收前次微批次梯度前即可继续处理后续输入数据。该方法通过计算与通信的重叠执行，在硬件效率（单次迭代耗时）上优于批量同步并行方案，但会引发梯度延迟问题，导致统计效率（达到目标精度所需迭代次数）下降[12,20]。

 Seideetal.[45,46]lookedatquantizinggradientstodecrease
 theamountofdataneededtobecommunicatedoverthenetwork.
 Thisapproximationstrategyiseffectiveforlimitedscenariosbut
 lacksgenerality; itdoesnothurtconvergenceforsomespeech
 models[47],buthasnotbeenshowntobeeffectiveforothertypes
 ofmodels.OthershaveexploredtechniquesfromtheHPClitera
turetoreducetheoverheadofcommunication[9,24,50,51],oftenusinghighlyspecializednetworkinghardware.Ourworkiscom
plementarytothesetechniquesandfocusesmainlyonimproving
 theperformanceofparallelDNNtrainingwhenusingcommodity
 acceleratorsandinterconnectsavailableinpublicclouds.
  acceleratorsandinterconnectsavailableinpublicclouds.翻译
Seide等人[45,46]提出了梯度量化方案以减少网络传输数据量。这种近似优化策略在特定场景下有效但缺乏普适性：虽然对某些语音模型不会影响收敛效果[47]，但尚未证明适用于其他类型模型。另有研究借鉴高性能计算领域的技术方案[9,24,50,51]来降低通信开销，这些方案通常依赖专用网络硬件。我们的工作与这些技术形成互补，主要致力于在公有云通用加速器与互联设备条件下提升并行DNN训练性能。

 Recentworkhasdemonstratedthatusinglargeminibatchesis
 effectivefortrainingResNet-50,especiallywhencombinedwith
 Layer-wiseAdaptiveRateScaling(LARS)[24,31,56].Largemini
batchesreducethecommunicationoverheadbyexchangingpa
rameterslessfrequently;however,ourexperimentsshowthatsuch
 techniqueslackgeneralitybeyondResNet-50andpipelineparal
lelismcanoutperformthefastestLARSdata-paralleloption.
近期研究表明，采用大型微批次结合层自适应速率缩放技术能有效训练ResNet-50模型[24,31,56]。大型微批次通过降低参数交换频率来减少通信开销，但我们的实验表明，此类技术对ResNet-50之外的其他模型缺乏普适性，而流水线并行方案的性能表现可超越最快的LARS数据并行方案。

ModelParallelism.Modelparallelismisanintra-batchparallelism
 approachwheretheoperators inaDNNmodelarepartitioned
 acrosstheavailableworkers,witheachworkerevaluatingand
 performingupdatesforonlyasubsetofthemodel’sparameters
 forall inputs.Theamountofdatacommunicatedisthesizeof
 intermediateoutputs(andcorrespondinggradients)thatneedto
 besentacrossworkers.
 模型并行作为批内并行的一种实现方式，其核心在于将深度神经网络中的运算单元划分至不同工作节点，每个节点仅针对所有输入数据计算并更新模型参数的特定子集。该方案下，通信数据量取决于需要在工作节点间传输的中间结果（及其对应梯度）的规模。

 Althoughmodelparallelismenablestrainingofverylargemod
els, vanillamodelparallelismisrarelyusedtoaccelerateDNN
 trainingbecauseitsuffersfromtwomajorlimitations.First,model
paralleltrainingresultsinunder-utilizationofcomputeresources,
 asillustratedinFigure2.Eachworkerisresponsibleforagroupof
 consecutivelayers; inthisregime,theintermediateoutputs(activa
tionsandgradients)betweenthesegroupsaretheonlydatathat
 needtobecommunicatedacrossworkers.2
 尽管模型并行技术能够支持超大规模模型的训练，但其原始形态却鲜少用于加速深度神经网络训练，主要存在两大局限性。首先，如图2所示，模型并行会导致计算资源利用率不足：每个工作节点负责处理连续层组时，仅需在节点间传输层组间的中间数据（激活值与梯度），这种机制使得大部分计算单元处于间歇性空闲状态。
<img width="415" height="259" alt="image" src="https://github.com/user-attachments/assets/46f4e2f0-d953-4cc9-adcc-0e369d46727b" />


 Thesecondlimitationformodel-parallel trainingisthat the
 burdenofpartitioningamodel acrossmultipleGPUs is left to
 theprogrammer[36], resultinginpointsolutions.Recentwork
 explorestheuseofReinforcementLearningtoautomaticallydeter
minedeviceplacementformodelparallelism[42].However,these
 techniquesaretime-andresource-intensive,anddonotleverage
 thefactthatDNNtrainingcanbethoughtofasacomputational
 pipelineconsistingofgroupsofconsecutivelayers–theseassump
tionsmaketheoptimizationproblemmoretractable,allowingfor
 exactsolutionsinpolynomialtimeasweshowin§3.1.
 模型并行训练的第二项局限在于，跨多GPU的模型划分工作完全依赖于人工实现[36]，导致解决方案高度定制化。近期研究尝试采用强化学习自动确定模型并行的设备布局[42]，但这类技术存在时间和资源消耗大的问题，且未能充分利用深度神经网络训练可视为连续层组构成计算流水线的特性——正如我们在3.1节所论证的，基于该特性可将优化问题转化为多项式时间内可解的规整形式。

  HybridIntra-batchParallelism.Recentworkhasproposedsplit
tingasingleiterationoftheoptimizationalgorithmamongmultiple
 dimensions.OWT[36]splitthethen-popularAlexNetmodelby
 hand,usingdataparallelismforconvolutional layersthathave
 asmallnumberofweightparametersandlargeoutputs,whilechoosingtonotreplicatefullyconnectedlayersthathavealarge
 numberofweightparametersandsmalloutputs.OWTdoesnot
 usepipelining.FlexFlow[33]proposedsplittingasingleiteration
 alongsamples,operators,attributes,andparameters,anddescribes
 analgorithmtodeterminehowtoperformthissplittinginanauto
matedway.However,FlexFlowdoesnotperformpipelining,and
 weshowinourexperiments(§5.3)thatthisleavesasmuchas90%
 ofperformanceonthetable.
 混合批内并行 的最新研究提出了沿多个维度拆分单次迭代的优化方案。OWT[36]通过人工方式对当时主流的AlexNet模型进行划分：对参数量少但输出量大的卷积层采用数据并行，而对参数量庞大的全连接层则不进行复制。该方案未采用流水线技术。FlexFlow[33]提出了沿样本、算子、特征和参数四个维度拆分单次迭代的方法，并设计了自动化划分算法。然而FlexFlow同样未实现流水线并行，我们的实验（§5.3）表明这种缺失会导致高达90%的性能潜力未被释放。

 2.2 Inter-batchParallelism

 Chenetal.[15]brieflyexploredthepotentialbenefitsofpipelin
ingminibatchesinmodel-paralleltraining,butdonotaddressthe
 conditionsforgoodstatisticalefficiency,scale,andgeneralityas
 applicabletolargereal-worldmodels.Huoetal.[29]exploredpar
allelizingthebackwardpassduringtraining.Ourproposedsolution
 parallelizesboththeforwardandbackwardpass.
 GPipe(concurrentworkwithanearlierPipeDreampreprint[25])
 usespipelininginthecontextofmodel-paralleltrainingforvery
 largemodels[28].GPipedoesnotspecifyanalgorithmforparti
tioningamodel,butassumesapartitionedmodelasinput.GPipe
 furthersplitsaminibatchintommicrobatches,andperformsfor
wardpassesfollowedbybackwardpassesforthesemmicrobatches
 (seeFigure3,m=4).Withafocusontrainingalargemodellike
 AmoebaNet,GPipeoptimizesformemoryefficiency;itusesexisting
 techniquessuchasweightgradientaggregationandtradescom
putationformemorybydiscardingactivationstashesbetweenthe
 forwardandthebackwardpass,insteadoptingtore-computethem
 whenneededinthebackwardpass[14].Asaresult, itcansuffer
 fromreducedhardwareefficiencyduetore-computationoverheads
 andfrequentpipelineflushesifmissmall(§5.4).
 Incomparison,PipeDreamaddresseskeyissuesignoredinprior
 work,offeringageneralsolutionthatkeepsworkerswellutilized,
 combiningpipeliningwithintra-batchparallelisminaprincipled
 way,whilealsoautomatingthepartitioningofthemodelacross
 theavailableworkers
Chen等人[15]曾简要探讨过在模型并行训练中采用微批次流水线处理的潜在优势，但未就统计效率、扩展性及普适性等关键条件进行深入分析，这些条件对于大型实际应用的模型至关重要。Huo等人[29]则研究了训练过程中反向传播的并行化。我们提出的解决方案实现了前向传播与反向传播的双重并行。
GPipe（与PipeDream预印本[25]前期研究同步的并行工作）在超大规模模型的模型并行训练中采用了流水线技术[28]。GPipe未提供模型划分的具体算法，而是以已划分的模型作为输入。该框架进一步将微批次划分为m个子微批次，并依次执行这些子微批次的前向传播与反向传播（如图3所示，m=4）。针对如AmoebaNet等大型模型的训练，GPipe以内存效率为优化目标：它采用权重梯度聚合等现有技术，并通过在前向与反向传播间丢弃激活值缓存（选择在反向传播需要时重新计算[14]），以计算资源换取内存空间。这导致当m值较小时，可能因重新计算开销和频繁的流水线刷新而降低硬件效率（参见5.4节）。
相较而言，PipeDream解决了前人工作中忽视的关键问题，提供了一种通用解决方案：通过原则性方法将流水线与批内并行相结合，保持工作节点的高效利用率，同时实现了模型在可用工作节点间的自动划分。
<img width="400" height="230" alt="image" src="https://github.com/user-attachments/assets/28baf95a-6bca-4b0d-bd03-1c4d1a9c4b1e" />

  2.3 DNNModelandHardwareDiversity

   DNNmodelsarediverse,withconvolutional layers,LSTMs[55],
 attentionlayers[53],andfully-connectedlayerscommonlyused.
 Thesedifferenttypesofmodelsexhibitvastlydifferentperformance
 characteristicswithdifferentparallelizationstrategies,makingthe
 optimalparallelizationstrategyhighlymodel-dependent.
 神经网络模型种类繁多，常包含卷积层、长短期记忆网络[55]、注意力层[53]以及全连接层等不同组件。这些不同类型的模型在使用不同并行化策略时，会表现出截然不同的性能特征，因此最优的并行化策略高度依赖于具体模型结构。

  Pickinganoptimalparallelizationschemeischallengingbecause
 theefficacyofsuchaschemedependsonthecharacteristicsof
 thetargetdeploymenthardwareaswell;GPUs,ASICs,andFPGAs
 haveverydifferentcomputecapabilities.Moreover, interconnects
 linkingtheseacceleratorshavedifferenttopologiesandcapacities;
 cloudserversarelinkedbytensto100Gbpsnetworks,accelera
torswithinserversmightbeconnectedoversharedPCIetrees(10
 to15GBps),andspecializedexpensiveservers,suchastheDGX
1[23],useNVLinkwithpoint-to-point30GBpsbandwidthcapabili
ties.Thisdiversityinmodelsanddeploymentsmakesitextremely
 hardtomanuallycomeupwithanoptimalparallelizationstrategy.
 PipeDreamautomatesthisprocess,aswediscussin§3.1.
选择最优并行化方案具有挑战性，因为方案的成效还取决于目标部署硬件的特性：GPU、ASIC和FPGA具有截然不同的计算能力。此外，连接这些加速器的互联网络也存在不同的拓扑结构与带宽容量——云服务器通过数十至100Gbps网络相连，服务器内部加速器可能通过共享PCIe树状总线连接（10至15GB/s），而像DGX-1[23]这类专业高端服务器则采用点对点NVLink技术，具备30GB/s的带宽能力。模型与部署环境的这种多样性，使得人工制定最优并行化策略变得极为困难。正如我们将在3.1节讨论的，PipeDream实现了这一过程的自动化。


 3 PIPELINEPARALLELISM

  PipeDreamusespipelineparallelism(PP), anewparallelization
 strategythatcombines intra-batchparallelismwithinter-batch
 parallelism.Pipeline-parallel computationinvolvespartitioning
 thelayersofaDNNmodelintomultiplestages,whereeachstage
 consistsofaconsecutivesetoflayersinthemodel.Eachstageis
 mappedtoaseparateGPUthatperformstheforwardpass(and
 backwardpass)foralllayersinthatstage.3
 PipeDream采用流水线并行这一新型并行策略，该策略将批内并行与批间并行相结合。流水线并行计算涉及将DNN模型的层划分成多个阶段，每个阶段由模型中连续的若干层组成。每个阶段被分配到独立的GPU上执行，由该GPU负责该阶段内所有层的前向传播（及反向传播）计算³。

 Inthesimplestcase,onlyoneminibatchisactiveinthesystem,
 asintraditionalmodel-paralleltraining(Figure2);inthissetup,at
 mostoneGPUisactiveatatime.Ideally,wewouldlikeallGPUsto
 beactive.Withthisinmind,weinjectmultipleminibatchesintothe
 pipelineoneaftertheother.Oncompletingitsforwardpassfora
 minibatch,eachstageasynchronouslysendstheoutputactivations to the next stage, while simultaneously starting to process another
 minibatch. The last stage starts the backward pass on a minibatch
 immediately after the forward pass completes. On completing its
 backward pass, each stage asynchronously sends the gradient to the
 previous stage while starting computation for the next minibatch
 (Figure 4).
 在最简单的情况下，如传统模型并行训练（图2）所示，系统中仅有一个微批次处于处理状态；在这种配置下，同一时间最多只有一个GPU处于活跃状态。理想情况下，我们希望所有GPU都能保持持续活跃。基于此目标，我们采用连续向流水线注入多个微批次的方式：当某个阶段完成对当前微批次的前向传播后，会异步将输出激活值传递给下一阶段，同时立即开始处理另一个微批次；最后一个阶段在前向传播结束后即刻启动该微批次的反向传播；而当每个阶段完成其反向传播计算时，会异步将梯度传递给前序阶段，同时开始处理下一个微批次的计算任务（图4）。
 <img width="400" height="243" alt="image" src="https://github.com/user-attachments/assets/fb6695f8-7f6d-4df7-ac82-423be1cc3018" />

 Pipeline parallelism canoutperformintra-batchparallelismmeth
ods for two reasons:
 Pipelining communicates less. PP often can communicate far
 less than DP. Instead of having to aggregate gradients for all param
eters and send the result to all workers, as is done in data-parallel
 approaches (using either collective communication or a parameter
 server), each worker in a PP execution has to communicate only
 subsets of the gradients and output activations, to only a single
 other worker. This can result in large reductions in communication
 for some models (e.g., >85% reduction for VGG-16, AWD LM).
 Pipelining overlaps computation and communication. Asyn
chronous communication of forward activations and backward
 gradients across stages results in significant overlap of communica
tion with the computation of a subsequent minibatch, as shown in
 Figure 5. This computation andcommunicationarecompletelyinde
pendent with no dependency edges, since they operate on different
 inputs, leading to easier parallelization.
 However, to realize the opportunity of PP, PipeDream must over
comethree challenges. In discussing PipeDream’s solutions to these
 challenges, we will refer to Figure 6, which shows PipeDream’s
 high-level workflow.流水线并行技术能够超越批内并行方法，主要基于两个关键原因：

流水线通信量更少
相比数据并行方法需要聚合所有参数的梯度并将结果发送给所有工作节点（无论采用集体通信或参数服务器架构），流水线并行中的每个工作节点仅需将部分梯度和输出激活值传递给单个相邻节点。这种通信模式可为某些模型大幅降低通信开销（例如VGG-16和AWD LM模型的通信量减少超过85%）。

流水线实现计算通信重叠
如图5所示，阶段间前向激活值与反向梯度的异步通信机制，使得通信操作与后续微批次的计算任务实现显著重叠。由于这些操作处理的是不同的输入数据，它们之间不存在依赖关系，这种完全独立性使得并行化更易实现。
<img width="424" height="485" alt="image" src="https://github.com/user-attachments/assets/b4fa17e5-fa7e-4a20-aefa-268b02c4a19e" />

然而，要充分发挥流水线并行的潜力，PipeDream必须解决三大挑战。在探讨其解决方案时，我们将参考图6所示的PipeDream高层工作流程。
<img width="402" height="383" alt="image" src="https://github.com/user-attachments/assets/8adafee3-95ba-47ed-8e1c-e22700838eae" />

 3.1 Challenge 1: Work Partitioning

  PipeDream treats model training as a computation pipeline, with
 each worker executing a subset of the model as a stage. Like with
 any pipeline, the steady state throughput of the resulting pipeline
 is the throughput of the slowest stage. Having each stage process
 minibatches at vastly different throughputs can lead to bubbles in the pipeline, starving faster stages of minibatches to work on and
 resulting in resource under-utilization. Excessive communication
 between workers can also lower the throughput of the training
 pipeline. Moreover, the allocation of stages to workers needs to
 be model- and hardware-aware to be effective, and there may be
 cases where no simple partitioning across the GPUs achieves both
 limited communication and perfect load balance.
 PipeDream将模型训练视为一个计算流水线，每个工作节点作为独立阶段执行模型的子集。与所有流水线系统类似，最终流水线的稳定状态吞吐量取决于最慢阶段的处理速度。当各阶段处理微批次的吞吐量差异过大时，会导致流水线中出现"气泡"现象，使快速阶段因缺乏待处理数据而空闲，进而造成资源利用率不足。此外，工作节点间过度的通信也会降低训练流水线的整体吞吐量。更重要的是，阶段分配到工作节点的过程需要同时考虑模型特性与硬件拓扑才能实现最优效果，在某些情况下可能无法找到既保证有限通信又实现完美负载均衡的简单GPU划分方案。

  Solution: PipeDream’s optimizer outputs a balanced pipeline.
 Its algorithm partitions DNN layers into stages such that each stage
 completes at roughly the same rate, while trying to minimize com
munication across workers in a topology-aware way (for example,
 large outputs should be sent over higher bandwidth links if possi
ble). To further improve load balancing, PipeDream goes beyond
 straight pipelines, allowing a stage to be replicated (i.e., data paral
lelism is used on the stage). This partitioning problem is equivalent
 to minimizing the time taken by the slowest stage of the pipeline,
 and has the optimal sub-problem property: a pipeline that maximizes
 throughput given a worker count is composed of sub-pipelines that
 maximize throughput for smaller worker counts. Consequently, we
 use dynamic programming to find the optimal solution.
 解决方案：PipeDream优化器可生成均衡的流水线配置。其算法将DNN层划分为若干阶段，确保每个阶段的处理速率基本一致，同时以拓扑感知的方式尽可能减少工作节点间的通信（例如，大型输出数据应优先通过高带宽链路传输）。为进一步优化负载均衡，PipeDream突破了传统直线流水线的限制，支持阶段复制（即在特定阶段采用数据并行技术）。该划分问题本质上等价于最小化流水线最慢阶段的处理时间，且具有最优子问题特性：在给定工作节点数条件下实现最大吞吐量的流水线，必然由较少数节点下实现最大吞吐量的子流水线构成。因此，我们采用动态规划算法来寻找最优划分方案。

  PipeDream exploits the fact that DNN training shows little vari
ance in computation time across inputs. PipeDream records the
 computation time taken by the forward and backward pass, the size
 of the layer outputs, and the size of the associated parameters for
 each layer as part of an initial profiling step; this profile is used as
 the input to the optimizer’s partitioning algorithm (Figure 6). The
 partitioning algorithm also takes into account other constraints
 such as hardware topology and bandwidth, number of workers, and
 memory capacity of the compute devices.

  Profiler. PipeDream records three quantities for each layer l, using
 a short (few minutes) profiling run of 1000 minibatches on a single
 GPU: 1) Tl, the total computation time across the forward and
 backward passes for layer l on the target GPU, 2) al, the size of the
 output activations of layer l (and the size of input gradients in the
 backward pass) in bytes, and 3) wl, the size of weight parameters
 for layer l in bytes.
 PipeDream利用了深度学习训练在不同输入间计算时间差异极小的特性。在初始性能分析阶段，系统会记录每个层的前向传播与反向传播计算时间、层输出数据大小及相关参数规模，这些分析数据将作为优化器划分算法的输入（图6）。该划分算法同时会综合考虑硬件拓扑结构、带宽资源、工作节点数量以及计算设备内存容量等其他约束条件。

  PipeDream estimates the communication time by dividing the
 amount of data that needs to be transferred by the network band
width of the communication link. Assuming efficient all_reduce
 collective communication, in data-parallel configurations with m
 workers, each worker sends (m−1
 m · |wl |) bytes to other workers,
 and receives the same amount; this is used to estimate the time
 for weight synchronization for layer l when using data parallelism
 withm workers.
 PipeDream通过待传输数据量除以通信链路的网络带宽来估算通信时间。在采用高效全归约集体通信的前提下，对于包含m个工作节点的数据并行配置，每个工作节点需要向其他节点发送（(m-1)/m · |wl|）字节的数据，并接收等量数据。该计算模型被用于估算在使用m个工作节点进行数据并行时，第l层的权重同步所需时间。

  Partitioning Algorithm. Our partitioning algorithm takes the
 output of the profiling step, and computes: 1) a partitioning of
 layers into stages, 2) the replication factor (number of workers) for
 each stage, and 3) optimal number of in-flight minibatches to keep
 the training pipeline busy.
 划分算法。我们的划分算法以性能分析数据作为输入，计算出三个关键要素：1）将网络层划分至各阶段的方案；2）每个阶段的复制因子（即工作节点数量）；3）维持训练流水线持续饱和所需的最优并行微批次数量。

  PipeDream’s optimizer assumes that the machine topology is
 hierarchical and can be organized into levels, as shown in Figure 7.
 Bandwidths within a level are the same, while bandwidths across
 levels are different. We assume that level k is comprised of mk
 components of level (k − 1), connected by links of bandwidth Bk.
 In Figure 7,m2 = 2 andm1 = 4. In addition, we definem0 to be 1;
m0representsthenumberofcomputedeviceswithinthefirstlevel
 (solidgreenboxesinFigure7).
 PipeDream优化器采用层级化机器拓扑结构，如图7所示，该结构可被组织为多个层级。同一层级内的带宽保持一致，而跨层级间的带宽则存在差异。我们假设第k层级由m_k个第(k-1)层组件构成，这些组件通过带宽为B_k的链路相互连接。在图7示例中，m_2=2且m_1=4。此外，我们定义m_0=1，其中m_0表示第一层级内的计算设备数量（对应图7中实线绿框所标示的单元）。

 PipeDream’soptimizersolvesdynamicprogrammingproblems
 progressivelyfromthelowesttothehighestlevel.Intuitively,this
 processfindstheoptimalpartitioningwithinaserverandthenuses
 thesepartitionstosplitamodeloptimallyacrossservers.
 Notation.LetAk(i→j,m)denotethetimetakenbytheslowest
 stageintheoptimalpipelinebetweenlayersiandjusingmworkers
 atlevelk.ThegoalofouralgorithmistofindAL(0→N,mL),and
 thecorrespondingpartitioning,whereListhehighestlevelandN
 isthetotalnumberoflayersinthemodel.
 LetTk(i→j,m)denotethetotaltimetakenbyasinglestage
 spanninglayersi throughj forbothforwardandbackwardpasses,
 replicatedovermworkersusingbandwidthBk.
 <img width="379" height="88" alt="image" src="https://github.com/user-attachments/assets/b9f11894-f138-4b92-a364-3778290222e2" />
PipeDream优化器采用自底向上的动态规划求解方法。直观来看，该过程首先在服务器内部寻找最优划分方案，继而利用这些划分实现跨服务器的模型最优分割。

符号体系
设Aₖ(i→j, m)表示在第k层级使用m个工作节点时，层i至层j间最优流水线中最慢阶段的执行时间。本算法的核心目标是求解A_L(0→N, m_L)及其对应划分方案，其中L代表最高层级，N表示模型总层数。
令Tₖ(i→j, m)表示跨越层i至层j的单一阶段在前向与反向传播中的总执行时间，该阶段通过带宽Bₖ在m个工作节点上进行复制。

wherethefirstterminsidethemaxisthetotalcomputationtime
 forallthelayersinthestageusinglevelk−1asthecomputation
 substrate,andthesecondtermisthetimefordata-parallelcom
municationamongall layersinthestage.Theresultofthemax
 expressionabovegivestheeffectivetimespentprocessingmin
putswhileperformingcomputeandcommunicationconcurrently;
 thus,theeffectivetimespentprocessingasingleinputisthisterm
 dividedbym.
 Theoptimalpipelinecannowbebrokenintoanoptimalsub
pipelineconsistingoflayersfrom1throughswithm−m′workers
 followedbyasinglestagewithlayerss+1throughj replicated
 overm′workers.Then,usingtheoptimalsub-problemproperty,
 wehave
 <img width="392" height="69" alt="image" src="https://github.com/user-attachments/assets/ac187dd9-8b97-4436-9961-f7db56e45419" />
其中，max函数内的首项代表该阶段所有层在以第k-1层级为计算基底时的总计算时间，次项则表示该阶段内所有层进行数据并行通信所需的时间。该max表达式的结果给出了在并发执行计算与通信时处理单个输入的有效时间，因此处理单个输入的实际有效时间需将此结果除以m。

根据最优子问题特性，最优流水线可拆分为两个部分：由层1至层s组成、使用m-m'个工作节点的最优子流水线，后接一个跨越层s+1至层j、在m'个工作节点上复制的独立阶段。由此可得：

  stageoftheoptimalsub-pipelinebetweenlayersiandswithm−m′
 workers, thesecondtermisthetimetakentocommunicatethe
 activationsandgradientsofsizeasbetweenlayerssands+1,and
 thethirdtermisthetimetakenbythesinglestagecontaininglayers
 s+1toj inadata-parallelconfigurationofm′workers.

Whensolvingforlevelk,weuseAk−1(i→j,mk−1),whichis
 theoptimal totalcomputationtimefor layersi throughjusing
 allworkersavailableinasinglecomponentat level (k−1) (in


