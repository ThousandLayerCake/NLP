# D Baseline Models 

**TFN.** The Tensor Fusion Network (TFN) (Zadeh et al., 2017) encodes three modalities with corresponding embedding subnetworks and uses outerproduct to model the unimodal, bimodal, and trimodal interactions as the fusion results. 

**LMF.** The Low-rank Multimodal Fusion (LMF) (Liu and Shen, 2018) utilizes low-rank tensors to improve efficiency of multimodal fusion. 

**MulT.** The Multimodal Transformer (MulT) (Tsai et al., 2019) proposes directional pairwise cross-modal attention that adapts one modality into another for multimodal fusion. 

**ICCN.** The Interaction Canonical Correlation Network (ICCN) (Sun et al., 2020) learns textbased audio and text-based video features by optimizing canonical loss. These features are concatenated with the text features for downstream classifiers such as logistic regression. 

**MISA.** The Modality-Invariant and -Specific Representations (MISA) (Hazarika et al., 2020) designs a multitask loss including task prediction loss, reconstruction loss, similarity loss, and difference loss to learn modality-invariant and modalityspecific representations. 

**MAG-BERT.** The Multimodal Adaptation Gate for Bert (MAG-BERT) (Rahman et al., 2020) builds an alignment gate that allows audio and video information to leak into the BERT model for multimodal fusion. 

**Self–MM.** The Self-Supervised Multitask Learning (Self–MM) (Yu et al., 2021) proposes a label generation module based on self-supervised learning to obtain unimodal supervision. Then they joint train the multimodal and unimodal tasks for better fusion results. 

**HyCon.** The Hybrid Contrastive Learning (HyCon) (Mai et al., 2021) performs intra- and inter-modal contrastive learning as well as semicontrastive learning within a modality to explore cross-modal interactions. 

**MMIM.** MultiModal InfoMax (MMIM) (Han et al., 2021) maximizes the mutual information in unimodal input pairs as well as between multimodal fusion result and unimodal input to aid the main MSA task.