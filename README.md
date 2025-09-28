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

 2 BACKGROUNDANDRELATEDWORK
 
 ADNNmodeliscomposedofmanyoperators organized into layers.
 When parallelizing DNN training, these layers may be partitioned
 over the available workers in different ways. In this section, we
 cover two broad classes of parallel DNN training: intra- and inter
batch. We also highlight the challenges posed by DNN model and
 hardware diversity for effective parallelization.

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

 The most commonly used form of data parallelism, referred to
 as bulk synchronous parallel or BSP [52]1, requires each worker to
 wait for gradients from other workers. Despite optimizations such
 as Wait-free Backpropagation [57], where weight gradients are sent
 as soon as they are available (common in modern frameworks),
 communication stalls are sometimes inevitable for large modelswherethetimeneededtosynchronizegradientsacrossworkers
 candominatecomputationtime.

  Figure1quantitativelyshowsthefractionoftrainingtimespent
 incommunicationstallswithdataparallelismfordifferentclassesof
 DNNsusingthreetypesofservers:8-1080TiGPUinstanceslinked
 overPCIewithinserversand25Gbpsinterconnectsacrossservers,
 4-V100GPUinstanceswithoutNVLinkand10Gbpsinterconnects
 acrossservers,and8-V100GPUinstanceswithNVLinkintercon
nectswithinserversand25Gbpsinterconnectsacrossservers.

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

 OtherDPOptimizations.Asynchronousparalleltraining(ASP)
 allowseachworkertoproceedwiththenextinputminibatchbe
forereceivingthegradients fromthepreviousminibatch.This
 approachimproveshardwareefficiency(timeneededperiteration)
 overBSPbyoverlappingcomputationwithcommunication,but
 alsointroducesstalenessandreducesstatisticalefficiency(number
 ofiterationsneededtoreachaparticulartargetaccuracy)[12,20].

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

 Recentworkhasdemonstratedthatusinglargeminibatchesis
 effectivefortrainingResNet-50,especiallywhencombinedwith
 Layer-wiseAdaptiveRateScaling(LARS)[24,31,56].Largemini
batchesreducethecommunicationoverheadbyexchangingpa
rameterslessfrequently;however,ourexperimentsshowthatsuch
 techniqueslackgeneralitybeyondResNet-50andpipelineparal
lelismcanoutperformthefastestLARSdata-paralleloption.

ModelParallelism.Modelparallelismisanintra-batchparallelism
 approachwheretheoperators inaDNNmodelarepartitioned
 acrosstheavailableworkers,witheachworkerevaluatingand
 performingupdatesforonlyasubsetofthemodel’sparameters
 forall inputs.Theamountofdatacommunicatedisthesizeof
 intermediateoutputs(andcorrespondinggradients)thatneedto
 besentacrossworkers.

 Althoughmodelparallelismenablestrainingofverylargemod
els, vanillamodelparallelismisrarelyusedtoaccelerateDNN
 trainingbecauseitsuffersfromtwomajorlimitations.First,model
paralleltrainingresultsinunder-utilizationofcomputeresources,
 asillustratedinFigure2.Eachworkerisresponsibleforagroupof
 consecutivelayers; inthisregime,theintermediateoutputs(activa
tionsandgradients)betweenthesegroupsaretheonlydatathat
 needtobecommunicatedacrossworkers.2

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
