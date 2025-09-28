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
 
 INTRODUCTION
 
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
 PipeDream takes a more nuanced approach to pipelining that
 outperforms other solutions– it achieves high hardware efficiency
 with no pipeline stalls in steady state, and high statistical efficiency
