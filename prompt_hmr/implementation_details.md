# 复现论文world coordinate evaluation metrics的实现细节

## 已有的代码
- vid_evaluator.py: 直接从evaluator复制过来的代码，你需要遵从evaluator的设计，从readme.md可以看到evaluator的调用方式，这一点上要大致保持一致
- vid_eval_utils.py：虽然PromptHMR没有实现这些全局的指标计算函数，但是它的pipeline的一部分：gvhmr有实现这些函数，因此我直接把PromptHMR/pipeline/gvhmr/hmr4d/utils/eval/eval_utils.py给复制了过来，你可以按需要进行修改、添补

## 一些简化
为了简化，也因为我的任务需要，我们只需要考虑EMDB数据集上的evaluation。因此你可能会想要阅读一下项目中对EMDB的读取和预处理，以及demo_video.py和相关文件、函数的实现方式，你需要在eval中先完整跑完这个video的重建，然后把结果进行eval。

为了你的方便，我已经整理了demo_video.py对一段视频做infer的结果的数据格式在result_format.md，你可以参考

在完成这个任务的过程中，我希望你保持一个进度追踪文档，放在PromptHMR/prompt_hmr下的progress.md中，记录一下你已完成的，还要todo，作为你的工作文档，因为可能一次完不成，需要交由其他开发者或程序员接手你的任务。



类别	指标名称	英文全称	含义 / 评估内容	单位
Camera-space Reconstruction	MPJPE	Mean Per Joint Position Error	pelvis 对齐后 3D 关节误差	mm
	PA-MPJPE	Procrustes Aligned MPJPE	Procrustes 对齐后 pose 误差（纯姿态）	mm
	PVE	Per Vertex Error	pose + shape 综合误差（网格顶点）	mm

World-space Motion	WA-MPJPE100	World-aligned MPJPE (100 frames)	对齐整段 100 帧世界系动作误差	mm
	W-MPJPE100	World MPJPE (100 frames)	仅对齐前 2 帧，衡量轨迹漂移	mm
	RTE	Root Trajectory Error	根部（pelvis）世界系轨迹误差（含 scale）	%
