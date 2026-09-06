import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

from vqc_core import seed_everything, load_or_generate_data, slice_partition, VQC_QNetwork, ReplayBuffer

st.set_page_config(page_title="ZN63 VQC-RL 故障诊断系统", layout="wide")

st.title("⚡ 高压真空断路器声纹变分量子强化学习 (VQC-RL) 智能诊断平台")
st.markdown("基于 4-Qubit 酉变换态矢演化网络与 MAX9814 声纹动力学特性的端到端检测系统")

# 侧边栏：参数配置与数据源
st.sidebar.header("⚙️ 诊断系统配置")
data_source = st.sidebar.radio("数据输入模式", ("内置物理仿真信号", "上传自定义 Excel 文件"))

norm_file, fault_file = None, None
if data_source == "上传自定义 Excel 文件":
    norm_file = st.sidebar.file_uploader("上传正常状态波形 (1_6.xlsx)", type=["xlsx"])
    fault_file = st.sidebar.file_uploader("上传故障状态波形 (1.1_3.xlsx)", type=["xlsx"])

st.sidebar.subheader("强化学习超参数")
epochs = st.sidebar.slider("训练轮数 (Epochs)", 10, 100, 30, step=5)
batch_size = st.sidebar.select_slider("批量大小 (Batch Size)", options=[8, 16, 32], value=16)
learning_rate = st.sidebar.number_input("AdamW 学习率", value=0.00015, format="%.5f")

tab1, tab2 = st.tabs(["📊 声学信号动力学分析", "🚀 模型训练与诊断评估"])

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预载入
norm_sigs, fault_sigs = load_or_generate_data(norm_file, fault_file, points_per_sample=30000, num_samples=20)

with tab1:
    st.subheader("时域声纹波形与傅里叶频域特性 (FFT)")
    col1, col2 = st.columns(2)

    t_axis = np.linspace(0, 0.6, 30000)
    fig_time, ax_t = plt.subplots(figsize=(7, 4))
    ax_t.plot(t_axis, norm_sigs[0], label="正常合闸", color="#1f77b4", alpha=0.8)
    ax_t.plot(t_axis, fault_sigs[0], label="机构受阻故障", color="#d62728", alpha=0.7)
    ax_t.set_title("时域波形 (0.6s)")
    ax_t.set_xlabel("时间 (s)")
    ax_t.set_ylabel("电压幅值 (V)")
    ax_t.legend()
    ax_t.grid(True, linestyle=":", alpha=0.6)
    col1.pyplot(fig_time)

    freqs = np.fft.rfftfreq(30000, 1.0 / 50000)
    fft_n = np.abs(np.fft.rfft(norm_sigs[0])) / 30000
    fft_f = np.abs(np.fft.rfft(fault_sigs[0])) / 30000
    fig_freq, ax_f = plt.subplots(figsize=(7, 4))
    ax_f.plot(freqs, fft_n, label="正常谱线", color="#1f77b4")
    ax_f.plot(freqs, fft_f, label="故障谱线", color="#d62728")
    ax_f.set_xlim(0, 8000)
    ax_f.set_title("FFT 频谱分布")
    ax_f.set_xlabel("频率 (Hz)")
    ax_f.set_ylabel("幅值")
    ax_f.legend()
    ax_f.grid(True, linestyle=":", alpha=0.6)
    col2.pyplot(fig_freq)

with tab2:
    if st.button("开始端到端训练与泛化验证", type="primary"):
        # 数据集划分与切片
        n_n, n_f = len(norm_sigs), len(fault_sigs)
        idx_n, idx_f = np.arange(n_n), np.arange(n_f)
        tr_n, te_n = train_test_split(idx_n, test_size=0.3, random_state=42)
        tr_f, te_f = train_test_split(idx_f, test_size=0.3, random_state=42)

        X_train, y_train = slice_partition(np.vstack([norm_sigs[tr_n], fault_sigs[tr_f]]),
                                           np.array([0] * len(tr_n) + [1] * len(tr_f)))
        X_test, y_test = slice_partition(np.vstack([norm_sigs[te_n], fault_sigs[te_f]]),
                                         np.array([0] * len(te_n) + [1] * len(te_f)))

        q_net = VQC_QNetwork(input_channels=1, num_actions=2).to(device)
        target_net = VQC_QNetwork(input_channels=1, num_actions=2).to(device)
        target_net.load_state_dict(q_net.state_dict())

        optimizer = optim.AdamW(q_net.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.SmoothL1Loss()
        buffer = ReplayBuffer(capacity=5000)

        gamma, epsilon, eps_min = 0.95, 0.90, 0.05
        eps_decay = (epsilon - eps_min) / (epochs * 0.7)

        progress_bar = st.progress(0)
        status_text = st.empty()
        metric_chart = st.empty()

        loss_curve, acc_curve = [], []

        for ep in range(1, epochs + 1):
            q_net.train()
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            ep_loss, correct, total = [], 0, 0

            for s_idx in range(0, len(indices), batch_size):
                b_idx = indices[s_idx: s_idx + batch_size]
                b_x = torch.tensor(X_train[b_idx], dtype=torch.float32).to(device)
                b_y = y_train[b_idx]

                with torch.no_grad():
                    q_vals = q_net(b_x)
                    actions = q_vals.argmax(dim=1).cpu().numpy()

                for i in range(len(actions)):
                    if np.random.rand() < epsilon:
                        actions[i] = np.random.randint(0, 2)

                rewards = np.where(actions == b_y, 1.0, -1.5)
                for s, a, r in zip(X_train[b_idx], actions, rewards):
                    buffer.push(s, a, r, s, True)

                correct += np.sum(actions == b_y)
                total += len(b_y)

                if len(buffer) >= batch_size:
                    s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size)
                    s_b, a_b, r_b, ns_b, d_b = s_b.to(device), a_b.to(device), r_b.to(device), ns_b.to(device), d_b.to(
                        device)
                    curr_q = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        max_next = target_net(ns_b).max(dim=1)[0]
                        target = r_b + gamma * max_next * (1.0 - d_b)
                    loss = criterion(curr_q, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ep_loss.append(loss.item())

            epsilon = max(eps_min, epsilon - eps_decay)
            with torch.no_grad():
                for p, tp in zip(q_net.parameters(), target_net.parameters()):
                    tp.data.copy_(0.15 * p.data + 0.85 * tp.data)

            loss_curve.append(np.mean(ep_loss) if ep_loss else 0.0)
            acc_curve.append(correct / total)
            progress_bar.progress(ep / epochs)
            status_text.text(f"训练进度: Epoch {ep}/{epochs} - 当前 Huber Loss: {loss_curve[-1]:.4f}")

        st.success("🎉 模型训练完成！正在测试集验证泛化指标...")

        # 测试集真实评估
        q_net.eval()
        t_preds, t_probs = [], []
        with torch.no_grad():
            for t_idx in range(0, len(X_test), batch_size):
                tx = torch.tensor(X_test[t_idx: t_idx + batch_size], dtype=torch.float32).to(device)
                t_q = q_net(tx)
                t_probs.extend(torch.softmax(t_q, dim=1)[:, 1].cpu().numpy())
                t_preds.extend(t_q.argmax(dim=1).cpu().numpy())

        acc = accuracy_score(y_test, t_preds)
        prec = precision_score(y_test, t_preds, zero_division=0)
        rec = recall_score(y_test, t_preds, zero_division=0)
        f1 = f1_score(y_test, t_preds, zero_division=0)

        # 核心指标卡片展示
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("准确率 (Accuracy)", f"{acc * 100:.2f}%")
        m2.metric("精确率 (Precision)", f"{prec * 100:.2f}%")
        m3.metric("召回率 (Recall)", f"{rec * 100:.2f}%")
        m4.metric("F1-Score", f"{f1:.4f}")

        # 可视化评估图表
        col_res1, col_res2 = st.columns(2)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        cm = confusion_matrix(y_test, t_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=['正常', '故障'],
                    yticklabels=['正常', '故障'])
        ax_cm.set_title("测试集混淆矩阵")
        col_res1.pyplot(fig_cm)

        fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
        fpr, tpr, _ = roc_curve(y_test, t_probs)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color='#d62728', lw=2, label=f"AUC = {roc_auc:.4f}")
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
        ax_roc.set_title("ROC 曲线")
        ax_roc.legend(loc="lower right")
        col_res2.pyplot(fig_roc)