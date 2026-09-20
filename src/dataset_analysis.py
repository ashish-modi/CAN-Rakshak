import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def dataset_analysis(dataset_path, cfg):

    print("Running Dataset Analysis...")
    file_name = cfg['file_name']
    file_path = os.path.join(dataset_path, "modified_dataset", file_name[:-3] + "csv")

    cols = ["timestamp","can_id","dlc",
            "b0","b1","b2","b3","b4","b5","b6","b7","flag"]

    df = pd.read_csv(file_path, header=None, names=cols)

    df["timestamp"] = df["timestamp"].astype(float)

    output_dir = os.path.join(dataset_path,"analysis", file_name[:-4])
    os.makedirs(output_dir, exist_ok=True)

    convert_payload_to_int(df)

    basic_statistics(df)

    plot_can_id_distribution(df, output_dir)

    plot_message_rate(df, output_dir)

    # plot_payload_histograms(df, output_dir)

    plot_canid_vs_time(df, output_dir)

    plot_payload_entropy(df, output_dir)

    plot_byte_correlation(df, output_dir)

    plot_canid_periodicity(df, output_dir)

    plot_attack_distribution(df, output_dir)

    print("Analysis saved to:", output_dir)



def convert_payload_to_int(df):

    payload_cols = ["b0","b1","b2","b3","b4","b5","b6","b7"]

    for col in payload_cols:
        df[col] = df[col].apply(lambda x: int(str(x),16))


def basic_statistics(df):

    print("\n===== Dataset Statistics =====")

    print("Total Frames:", len(df))
    print("Unique CAN IDs:", df["can_id"].nunique())
    print("DLC Distribution:\n", df["dlc"].value_counts())


def count_attack_scenarios(df):
    """Count contiguous blocks of attack messages."""
    is_attack = ~df["flag"].astype(str).str.strip().isin(["R", "0", "Normal"])
    transitions = is_attack & ~is_attack.shift(1, fill_value=False)
    return int(transitions.sum())


def get_attack_ids(df):
    """Return CAN IDs whose traffic is >50 % attack messages."""
    is_attack = ~df["flag"].astype(str).str.strip().isin(["R", "0", "Normal"])
    attack_counts = df.loc[is_attack,  "can_id"].value_counts()
    normal_counts = df.loc[~is_attack, "can_id"].value_counts()
    result = {}
    for cid, cnt in attack_counts.items():
        norm = int(normal_counts.get(cid, 0))
        total = cnt + norm
        ratio = cnt / total if total else 0
        if ratio > 0.5:
            result[cid] = {"attack_msgs": int(cnt), "normal_msgs": norm,
                           "attack_ratio": round(float(ratio), 4)}
    return result


def save_statistics(df, output_dir, file_name, attack_type=None):
    """Write per-file statistical report to <output_dir>/statistics.txt."""
    SEP  = "=" * 70
    sep2 = "-" * 50

    total      = len(df)
    unique_ids = df["can_id"].nunique()
    all_ids    = sorted(df["can_id"].unique(), key=lambda x: int(str(x), 16)
                        if all(c in "0123456789abcdefABCDEF" for c in str(x)) else 0)
    dlc_dist   = df["dlc"].value_counts().sort_index()

    is_attack   = ~df["flag"].astype(str).str.strip().isin(["R", "0", "Normal"])
    n_normal    = int((~is_attack).sum())
    n_attack    = int(is_attack.sum())
    attack_ratio = n_attack / total if total else 0

    flag_counts = df["flag"].value_counts().to_dict()
    n_scenarios = count_attack_scenarios(df)
    atk_ids     = get_attack_ids(df)

    lines = []
    def w(*args):
        lines.append(" ".join(str(a) for a in args))

    w(SEP)
    w(f"  STATISTICAL ANALYSIS REPORT")
    w(f"  File : {file_name}")
    if attack_type:
        w(f"  Attack type : {attack_type}")
    w(SEP)
    w()

    w("1. TOTAL LOGS")
    w(sep2)
    w(f"   Total messages     : {total:,}")
    w(f"   Normal messages    : {n_normal:,}  ({n_normal/total*100:.2f}%)" if total else "")
    w(f"   Attack messages    : {n_attack:,}  ({attack_ratio*100:.2f}%)" if total else "")
    w()

    w("2. UNIQUE CAN IDs")
    w(sep2)
    w(f"   Count : {unique_ids}")
    w(f"   IDs   : {', '.join(str(i).upper() for i in all_ids)}")
    w()

    w("3. DLC DISTRIBUTION")
    w(sep2)
    for dlc_val, cnt in dlc_dist.items():
        w(f"   DLC={int(dlc_val)}  →  {cnt:>10,}  ({cnt/total*100:5.2f}%)")
    w()

    w("4. NUMBER OF MESSAGES")
    w(sep2)
    w(f"   Total   : {total:,}")
    w(f"   Normal  : {n_normal:,}")
    w(f"   Attack  : {n_attack:,}")
    w()

    w("5. LABEL / FLAG DISTRIBUTION")
    w(sep2)
    for lbl, cnt in sorted(flag_counts.items(), key=lambda x: -x[1]):
        w(f"   '{lbl}'  →  {cnt:>10,}  ({cnt/total*100:5.2f}%)")
    w()


    w("6. CAN IDs USED FOR ATTACK  (attack ratio > 50%)")
    w(sep2)
    if atk_ids:
        for cid, info in sorted(atk_ids.items()):
            w(f"   CAN ID {cid.upper():>8}  |  "
              f"attack: {info['attack_msgs']:>8,}  |  "
              f"normal: {info['normal_msgs']:>8,}  |  "
              f"ratio: {info['attack_ratio']*100:5.2f}%")
    else:
        w("   (none — this file contains only normal traffic)")
    w()

    w(SEP)
    w()

    out_path = os.path.join(output_dir, "statistics.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Statistics saved to: {out_path}")


def plot_can_id_distribution(df, output_dir):

    counts = df["can_id"].value_counts().head(20)

    plt.figure()
    counts.plot(kind="bar")
    plt.title("Top 20 CAN ID Frequency")
    plt.xlabel("CAN ID")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"canid_distribution.png"))
    plt.close()



def plot_message_rate(df, output_dir):
    bin_size = 100.0  # 10-second bins
    t0 = df["timestamp"].min()
    rel = df["timestamp"] - t0
    bins = (rel // bin_size) * bin_size

    normal_mask = df["flag"].astype(str).str.strip().isin(["R", "0", "Normal"])
    all_bins = np.arange(0, bins.max() + bin_size, bin_size)

    normal_rate = df[normal_mask].groupby(bins[normal_mask]).size().reindex(all_bins, fill_value=0)
    attack_rate = df[~normal_mask].groupby(bins[~normal_mask]).size().reindex(all_bins, fill_value=0)
    total_rate  = normal_rate + attack_rate

    n_normal = int(normal_mask.sum())
    n_attack = int((~normal_mask).sum())
    has_attack = attack_rate.max() > 0

    # One bar per bin; bar centres at all_bins, width slightly less than bin_size
    width = bin_size * 0.85
    n_bins = len(all_bins)
    fig_width = max(24, n_bins * 0.09)
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    ax.bar(all_bins, normal_rate.values, width=width,
           color="#4C9BE8", alpha=0.85, label=f"Normal  ({n_normal:,})", zorder=2)
    if has_attack:
        ax.bar(all_bins, attack_rate.values, width=width,
               bottom=normal_rate.values,
               color="#E84C4C", alpha=0.85, label=f"Attack  ({n_attack:,})", zorder=2)

    # Label above each bar: benign count on top of the normal segment,
    # attack count on top of the total bar (only where attack > 0)
    for x, n, a, tot in zip(all_bins, normal_rate.values, attack_rate.values, total_rate.values):
        if tot == 0:
            continue
        if a > 0:
            # Two-line label: benign / attack
            ax.text(x, tot, f"B:{int(n)}\nA:{int(a)}",
                    ha="center", va="bottom", fontsize=4.5,
                    color="#111111", fontweight="bold", linespacing=1.2)
        else:
            ax.text(x, tot, f"{int(n)}",
                    ha="center", va="bottom", fontsize=4.5, color="#2176AE")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("No. of Messages", fontsize=12)
    ax.set_title("CAN Bus Message Rate (100-second bins)", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlim(all_bins[0] - bin_size, all_bins[-1] + bin_size)
    ax.set_ylim(bottom=0, top=total_rate.max() * 1.18)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, color="#cccccc")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "message_rate.png"), dpi=150, bbox_inches="tight")
    plt.close()




# def plot_payload_histograms(df, output_dir):

#     payload_cols = ["b0","b1","b2","b3","b4","b5","b6","b7"]

#     for col in payload_cols:

#         plt.figure()
#         df[col].hist(bins=50)

#         plt.title(f"{col} Value Distribution")
#         plt.xlabel("Value")
#         plt.ylabel("Count")

#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir,f"{col}_hist.png"))
#         plt.close()



def plot_canid_vs_time(df, output_dir):

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    unique_ids = sorted(df_sorted["can_id"].unique(), key=lambda x: int(x, 16))
    id_to_y = {cid: i for i, cid in enumerate(unique_ids)}
    n_ids = len(unique_ids)

    # Sample each class independently so both are always visible
    max_per_class = 200_000
    normal_mask = df_sorted["flag"].astype(str).str.strip().isin(["R", "0", "Normal"])
    normal_df = df_sorted[normal_mask]
    attack_df = df_sorted[~normal_mask]
    if len(normal_df) > max_per_class:
        normal_df = normal_df.sample(max_per_class, random_state=42)
    if len(attack_df) > max_per_class:
        attack_df = attack_df.sample(max_per_class, random_state=42)
    df_plot = pd.concat([normal_df, attack_df]).sort_values("timestamp")

    cmap = plt.colormaps.get_cmap("hsv")
    id_colors = {cid: cmap(i / n_ids) for i, cid in enumerate(unique_ids)}

    flag_str = df_plot["flag"].astype(str).str.strip()
    is_attack = ~flag_str.isin(["R", "0", "Normal"])

    fig_height = max(8, n_ids * 0.35)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    for cid in unique_ids:
        mask = df_plot["can_id"] == cid
        y_cid = id_to_y[cid]
        color = id_colors[cid]
        normal_mask = mask & ~is_attack
        if normal_mask.any():
            ax.scatter(
                df_plot.loc[normal_mask, "timestamp"],
                np.full(normal_mask.sum(), y_cid),
                s=1, alpha=0.4, color=color, linewidths=0, rasterized=True,
            )
        attack_mask = mask & is_attack
        if attack_mask.any():
            ax.scatter(
                df_plot.loc[attack_mask, "timestamp"],
                np.full(attack_mask.sum(), y_cid),
                s=8, alpha=1.0, color=color,
                edgecolors="black", linewidths=0.4, rasterized=True,
            )

    ax.set_yticks(range(n_ids))
    ax.set_yticklabels(unique_ids, fontsize=max(5, min(9, 180 // n_ids)))
    for label, cid in zip(ax.get_yticklabels(), unique_ids):
        label.set_color(id_colors[cid])
    ax.set_xlabel("Timestamp (s)", fontsize=11)
    ax.set_ylabel("CAN ID (hex)", fontsize=11)
    ax.set_title("CAN ID Occurrence Pattern Over Time", fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "canid_vs_time.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Zoomed view: 2-second window around attack onset (or start if no attacks)
    zoom_duration = 2.0
    if is_attack.any():
        t_attack_start = df_plot.loc[is_attack, "timestamp"].min()
        t_zoom_start = max(df_sorted["timestamp"].min(), t_attack_start - 1.0)
    else:
        t_zoom_start = df_sorted["timestamp"].min()
    t_zoom_end = t_zoom_start + zoom_duration

    zoom_mask = (df_sorted["timestamp"] >= t_zoom_start) & (df_sorted["timestamp"] <= t_zoom_end)
    df_zoom = df_sorted[zoom_mask]

    if len(df_zoom) == 0:
        return

    flag_zoom = df_zoom["flag"].astype(str).str.strip()
    is_attack_zoom = ~flag_zoom.isin(["R", "0", "Normal"])

    fig2, ax2 = plt.subplots(figsize=(16, fig_height))

    for cid in unique_ids:
        mask = df_zoom["can_id"] == cid
        y_cid = id_to_y[cid]
        color = id_colors[cid]
        normal_mask = mask & ~is_attack_zoom
        if normal_mask.any():
            ax2.scatter(
                df_zoom.loc[normal_mask, "timestamp"],
                np.full(normal_mask.sum(), y_cid),
                s=6, alpha=0.7, color=color, linewidths=0, rasterized=True,
            )
        attack_mask = mask & is_attack_zoom
        if attack_mask.any():
            ax2.scatter(
                df_zoom.loc[attack_mask, "timestamp"],
                np.full(attack_mask.sum(), y_cid),
                s=18, alpha=1.0, color=color,
                edgecolors="black", linewidths=0.5, rasterized=True,
            )

    ax2.set_yticks(range(n_ids))
    ax2.set_yticklabels(unique_ids, fontsize=max(5, min(9, 180 // n_ids)))
    for label, cid in zip(ax2.get_yticklabels(), unique_ids):
        label.set_color(id_colors[cid])
    ax2.set_xlabel("Timestamp (s)", fontsize=11)
    ax2.set_ylabel("CAN ID (hex)", fontsize=11)
    ax2.set_title(f"CAN ID Occurrence Pattern — {zoom_duration}s window "
                  f"[{t_zoom_start:.3f}s – {t_zoom_end:.3f}s]", fontsize=13)
    ax2.set_xlim(t_zoom_start, t_zoom_end)
    ax2.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "canid_vs_time_zoom.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_canid_periodicity(df, output_dir):

    df_sorted = df.sort_values("timestamp")

    unique_ids = sorted(df_sorted["can_id"].unique(), key=lambda x: int(x, 16))
    n_ids = len(unique_ids)

    # Compute inter-arrival times (ms) per CAN ID
    iat_per_id = []
    valid_ids  = []
    for cid in unique_ids:
        iat = df_sorted[df_sorted["can_id"] == cid]["timestamp"].diff().dropna() * 1000
        if len(iat) >= 2:
            iat_per_id.append(iat.values)
            valid_ids.append(cid)

    n_valid   = len(valid_ids)
    cmap      = plt.colormaps.get_cmap("hsv")
    id_colors = [cmap(i / n_ids) for i, cid in enumerate(unique_ids) if cid in valid_ids]

    medians = np.array([np.median(v) for v in iat_per_id])
    q3s     = np.array([np.percentile(v, 75) for v in iat_per_id])
    whisker_tops = np.array([np.percentile(v, 95) for v in iat_per_id])

    # Clip to 1.5× the median of all per-ID medians — keeps the common cluster
    # visible and pushes slow/fast outliers out to arrow annotations
    y_max = np.median(medians) * 2.5
    y_min = 0

    fig_width = max(10, n_valid * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    bp = ax.boxplot(
        iat_per_id,
        vert=True,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="#555555", linewidth=1.2),
        capprops=dict(color="#555555", linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )

    for patch, color in zip(bp["boxes"], id_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    # Median label above each box (skip if box is above the clipped window)
    for i, (med, q3) in enumerate(zip(medians, q3s), start=1):
        if med <= y_max:
            ax.text(i, q3, f"{med:.1f}", va="bottom", ha="center",
                    fontsize=7, color="black", fontweight="bold")

    # Arrow annotations for out-of-range IDs
    for i, med in enumerate(medians, start=1):
        if med > y_max:
            ax.annotate(f"↑ {med:.0f} ms", xy=(i, y_max), fontsize=7,
                        ha="center", va="top", color="black", fontweight="bold")
        elif med < y_min and y_min > 0:
            ax.annotate(f"↓ {med:.1f} ms", xy=(i, y_min), fontsize=7,
                        ha="center", va="bottom", color="black", fontweight="bold")

    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(1, n_valid + 1))
    ax.set_xticklabels(valid_ids, rotation=45, ha="right",
                       fontsize=max(5, min(9, 180 // n_valid)))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g} ms"))
    ax.set_ylabel("Inter-arrival Time (ms)", fontsize=11)
    ax.set_xlabel("CAN ID (hex)", fontsize=11)
    ax.set_title("CAN ID Periodicity — IQR of Inter-arrival Time",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, color="#cccccc")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "canid_periodicity.png"), dpi=150, bbox_inches="tight")
    plt.close()



def plot_payload_entropy_old(df, output_dir):

    payload_cols = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]

    def _msg_entropy(row):
        counts = np.bincount(row.astype(int), minlength=256).astype(float)
        probs  = counts / counts.sum()
        probs  = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    normal_mask = df["flag"].astype(str).str.strip().isin(["R", "0", "Normal"])
    payload_all = df[payload_cols].values
    idx        = np.random.default_rng(42).choice(len(df), min(80_000, len(df)), replace=False)
    ent_vals   = np.array([_msg_entropy(payload_all[i]) for i in idx])
    ent_normal = ent_vals[normal_mask.values[idx]]
    ent_attack = ent_vals[~normal_mask.values[idx]]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    bins = np.linspace(0, 8, 80)
    counts_n, bins_n, _ = ax.hist(ent_normal, bins=bins, density=True, alpha=0.55,
            color="#4C9BE8", label=f"Normal  (n={len(ent_normal):,})")
    counts_a, bins_a, _ = ax.hist(ent_attack, bins=bins, density=True, alpha=0.65,
            color="#E84C4C", label=f"Attack  (n={len(ent_attack):,})")

    if len(ent_normal):
        ax.axvline(ent_normal.mean(), color="#2176AE", linewidth=1.8,
                   linestyle="--", label=f"Normal mean  {ent_normal.mean():.2f} b")
    if len(ent_attack):
        ax.axvline(ent_attack.mean(), color="#B22222", linewidth=1.8,
                   linestyle="--", label=f"Attack mean  {ent_attack.mean():.2f} b")
        
    for i in range(len(counts_n)):
        if counts_n[i] > 0:  # avoid clutter
            x = (bins_n[i] + bins_n[i+1]) / 2
            y = counts_n[i]
            ax.text(x, y, f"({x:.2f}, {y:.3f})",
                    ha='center', va='bottom', fontsize=6, rotation=90)
            
    for i in range(len(counts_a)):
        if counts_a[i] > 0:
            x = (bins_a[i] + bins_a[i+1]) / 2
            y = counts_a[i]
            ax.text(x, y, f"({x:.2f}, {y:.3f})",
                    ha='center', va='bottom', fontsize=6, rotation=90, color='darkred')

    ax.set_xlabel("Per-message Entropy (bits)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Per-message Payload Entropy Distribution — Normal vs Attack",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, color="#cccccc")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "payload_entropy.png"), dpi=150, bbox_inches="tight")
    plt.close()

def plot_payload_entropy(df, output_dir):


    payload_cols = ["b0","b1","b2","b3","b4","b5","b6","b7"]

    def _msg_entropy(row):
        counts = np.bincount(row.astype(int), minlength=256).astype(float)
        probs  = counts / counts.sum()
        probs  = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    # ---- Data prep ----
    normal_mask = df["flag"].astype(str).str.strip().isin(["R","0","Normal"])
    payload_all = df[payload_cols].values

    idx = np.random.default_rng(42).choice(len(df), min(80_000, len(df)), replace=False)
    ent_vals   = np.array([_msg_entropy(payload_all[i]) for i in idx])
    ent_normal = ent_vals[normal_mask.values[idx]]
    ent_attack = ent_vals[~normal_mask.values[idx]]

    # ---- Plot ----
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    # Dynamic bins
    x_max = max(ent_vals) + 0.3
    bins = np.linspace(0, x_max, 60)

    # Histograms (light)
    ax.hist(ent_normal, bins=bins, density=True,
            alpha=0.3, color="#4C9BE8")

    ax.hist(ent_attack, bins=bins, density=True,
            alpha=0.3, color="#E84C4C")

    # KDE curves (main visual)
    sns.kdeplot(ent_normal, ax=ax, color="#2176AE", linewidth=2.2,
                label=f"Normal (n={len(ent_normal):,})")

    sns.kdeplot(ent_attack, ax=ax, color="#B22222", linewidth=2.2,
                label=f"Attack (n={len(ent_attack):,})")

    # Means
    if len(ent_normal):
        ax.axvline(ent_normal.mean(), color="#2176AE",
                   linestyle="--", linewidth=1.5)

    if len(ent_attack):
        ax.axvline(ent_attack.mean(), color="#B22222",
                   linestyle="--", linewidth=1.5)

    # Labels
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Per-message Entropy (bits)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Payload Entropy Distribution: Normal vs Attack",
                 fontsize=13, fontweight="bold")

    ax.legend(frameon=True, fontsize=9)
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "payload_entropy_clean.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

def plot_byte_correlation(df, output_dir):

    payload_cols = ["b0","b1","b2","b3","b4","b5","b6","b7"]

    corr = df[payload_cols].corr()

    plt.figure()
    plt.imshow(corr)
    plt.colorbar()

    plt.xticks(range(8), payload_cols)
    plt.yticks(range(8), payload_cols)

    plt.title("Payload Byte Correlation")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"byte_correlation.png"))
    plt.close()


def plot_attack_distribution(df, output_dir):

    if "label" not in df.columns:
        print("No label column found. Skipping attack distribution.")
        return

    counts = df["label"].value_counts()

    labels = ["Benign", "Attack"]

    plt.figure()

    plt.bar(labels, counts)

    plt.title("Attack vs Benign Distribution")
    plt.ylabel("Number of Frames")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attack_distribution.png"))
    plt.close()


def plot_class_distribution(df, output_dir):
    print("Plotting class distribution...")

    plt.figure(figsize=(6,4))

    df["flag"].value_counts().plot(kind="bar")

    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Messages")
    plt.xticks([0,1],["Normal","Attack"], rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()

def plot_timeline_distribution(df, output_dir):

    t0           = df["timestamp"].min()
    rel          = df["timestamp"] - t0
    normal_mask  = df["flag"].astype(str).str.strip().isin(["R", "0", "Normal"])
    n_bins       = 200
    bins         = np.linspace(0, rel.max(), n_bins + 1)
    bin_centres  = (bins[:-1] + bins[1:]) / 2

    n_normal, _  = np.histogram(rel[normal_mask],  bins=bins)
    n_attack, _  = np.histogram(rel[~normal_mask], bins=bins)
    n_total      = n_normal + n_attack

    # Attack proportion per bin (for the bottom panel)
    with np.errstate(invalid="ignore"):
        atk_pct = np.where(n_total > 0, n_attack / n_total * 100, 0.0)

    # Cumulative message count
    cumulative = np.cumsum(n_total)

    has_attack  = n_attack.sum() > 0
    atk_bins    = bin_centres[n_attack > 0]
    t_atk_start = atk_bins[0]  if has_attack else None
    t_atk_end   = atk_bins[-1] if has_attack else None

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1.2], "hspace": 0.08},
    )
    fig.patch.set_facecolor("#f8f9fa")
    for ax in (ax_top, ax_mid, ax_bot):
        ax.set_facecolor("#f8f9fa")

    # ── Top: stacked message count ─────────────────────────────────────────
    ax_top.fill_between(bin_centres, n_normal,
                        step="mid", alpha=0.6, color="#4C9BE8", label="Normal")
    ax_top.plot(bin_centres, n_normal,
                drawstyle="steps-mid", color="#2176AE", linewidth=0.7, alpha=0.8)
    if has_attack:
        ax_top.fill_between(bin_centres, n_normal, n_total,
                            step="mid", alpha=0.7, color="#E84C4C", label="Attack")
        ax_top.plot(bin_centres, n_total,
                    drawstyle="steps-mid", color="#B22222", linewidth=0.7, alpha=0.8)
        ax_top.axvspan(t_atk_start, t_atk_end, color="#E84C4C", alpha=0.07, zorder=0)
        ax_top.axvline(t_atk_start, color="#B22222", linewidth=1.4,
                       linestyle="--", alpha=0.85)
        ax_top.annotate(
            f"Attack start\n{t_atk_start:.1f}s",
            xy=(t_atk_start, ax_top.get_ylim()[1]),
            xytext=(t_atk_start + (rel.max() * 0.01), n_total.max() * 0.92),
            fontsize=8, color="#B22222", fontweight="bold",
        )
    total_msgs  = len(df)
    attack_pct  = (~normal_mask).sum() / total_msgs * 100
    ax_top.text(0.99, 0.97,
                f"Total: {total_msgs:,}  |  Attack: {attack_pct:.1f}%",
                transform=ax_top.transAxes, ha="right", va="top",
                fontsize=9, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax_top.set_ylabel("Messages / bin", fontsize=11)
    ax_top.set_title("CAN Bus Message Timeline Distribution", fontsize=13,
                     fontweight="bold", pad=10)
    ax_top.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax_top.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4, color="#cccccc")
    ax_top.spines[["top", "right"]].set_visible(False)

    # ── Middle: attack proportion ──────────────────────────────────────────
    ax_mid.fill_between(bin_centres, atk_pct,
                        step="mid", alpha=0.65, color="#E84C4C")
    ax_mid.plot(bin_centres, atk_pct,
                drawstyle="steps-mid", color="#B22222", linewidth=0.8)
    if has_attack:
        ax_mid.axvspan(t_atk_start, t_atk_end, color="#E84C4C", alpha=0.07, zorder=0)
    ax_mid.set_ylim(0, 105)
    ax_mid.set_ylabel("Attack %", fontsize=10)
    ax_mid.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_mid.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4, color="#cccccc")
    ax_mid.spines[["top", "right"]].set_visible(False)

    # ── Bottom: cumulative message count ───────────────────────────────────
    ax_bot.plot(bin_centres, cumulative, color="#2176AE", linewidth=1.8)
    ax_bot.fill_between(bin_centres, cumulative, alpha=0.2, color="#4C9BE8")
    if has_attack:
        ax_bot.axvspan(t_atk_start, t_atk_end, color="#E84C4C", alpha=0.07, zorder=0)
    ax_bot.set_ylabel("Cumulative\nmessages", fontsize=10)
    ax_bot.set_xlabel("Time (s)", fontsize=11)
    ax_bot.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K")
    )
    ax_bot.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4, color="#cccccc")
    ax_bot.spines[["top", "right"]].set_visible(False)

    plt.savefig(os.path.join(output_dir, "timeline_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_label_vs_time_windows(df, output_dir):
    """
    Three-panel scatter plot showing attack (red, y=1) vs benign (green, y=0)
    labels over time for three representative windows: beginning, middle, end.
    A full-dataset overview strip at the top shows global attack distribution
    and marks where each window sits.
    """
    t0        = df["timestamp"].min()
    t_max     = df["timestamp"].max()
    total_dur = t_max - t0

    rel_t     = df["timestamp"] - t0
    is_attack = ~df["flag"].astype(str).str.strip().isin(["R", "0", "Normal"])

    # Adaptive window width: 1/20th of total duration, clamped to [3 s, 30 s]
    win = float(np.clip(total_dur / 20, 3, 30))

    windows = [
        ("Beginning", 0.0,                  win),
        ("Middle",    total_dur / 2 - win / 2, total_dur / 2 + win / 2),
        ("End",       total_dur - win,       total_dur),
    ]
    accent = ["#2980b9", "#d35400", "#8e44ad"]   # blue, orange, purple

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("#f5f5f5")
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.5], hspace=0.52, wspace=0.30)

    ax_ov = fig.add_subplot(gs[0, :])
    axes  = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # ── Top: full-dataset message-density overview ─────────────────────────────
    n_bins = 400
    bins   = np.linspace(0, total_dur, n_bins + 1)
    bc     = (bins[:-1] + bins[1:]) / 2
    n_b, _ = np.histogram(rel_t[~is_attack], bins=bins)
    n_a, _ = np.histogram(rel_t[ is_attack], bins=bins)
    y_top  = int((n_b + n_a).max())

    ax_ov.fill_between(bc, n_b,          step="mid", color="#27ae60", alpha=0.75,
                       label=f"Benign  ({(~is_attack).sum():,})")
    ax_ov.fill_between(bc, n_b, n_b + n_a, step="mid", color="#e74c3c", alpha=0.85,
                       label=f"Attack  ({is_attack.sum():,})")

    for i, (name, t_s, t_e) in enumerate(windows):
        ax_ov.axvspan(t_s, t_e, color=accent[i], alpha=0.14, zorder=3)
        for t in (t_s, t_e):
            ax_ov.axvline(t, color=accent[i], lw=1.2, ls="--", alpha=0.75, zorder=4)
        ax_ov.text((t_s + t_e) / 2, y_top * 1.04, name,
                   ha="center", va="bottom", fontsize=9,
                   color=accent[i], fontweight="bold")

    ax_ov.set_xlim(0, total_dur)
    ax_ov.set_ylim(0, y_top * 1.22)
    ax_ov.set_xlabel("Time (s from start)", fontsize=10)
    ax_ov.set_ylabel("Messages / bin", fontsize=10)
    ax_ov.set_title(
        f"Full Dataset Overview  ·  Duration: {total_dur:.1f} s  ·  "
        f"Benign: {(~is_attack).sum():,}  ·  Attack: {is_attack.sum():,}",
        fontsize=11, fontweight="bold",
    )
    ax_ov.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_ov.spines[["top", "right"]].set_visible(False)
    ax_ov.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    ax_ov.set_facecolor("#f5f5f5")

    # ── Bottom: three zoomed scatter windows ───────────────────────────────────
    for ax, (name, t_s, t_e), col in zip(axes, windows, accent):
        mask  = (rel_t >= t_s) & (rel_t <= t_e)
        w_rel = rel_t[mask].values
        w_atk = is_attack[mask].values

        n_b_w = (~w_atk).sum()
        n_a_w =   w_atk.sum()

        # Alpha scales so dense windows stay readable (more dots → more transparent)
        alpha_b = float(np.clip(1500 / max(n_b_w, 1), 0.04, 1.0))
        alpha_a = float(np.clip(400  / max(n_a_w, 1), 0.35, 1.0))

        if n_b_w:
            ax.scatter(w_rel[~w_atk], np.zeros(n_b_w),
                       s=2, color="#27ae60", alpha=alpha_b,
                       linewidths=0, rasterized=True, label=f"Benign  ({n_b_w:,})")
        if n_a_w:
            ax.scatter(w_rel[w_atk], np.ones(n_a_w),
                       s=14, color="#e74c3c", alpha=alpha_a,
                       linewidths=0, rasterized=True, label=f"Attack  ({n_a_w:,})")

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Benign (0)", "Attack (1)"], fontsize=10)
        for tick, color in zip(ax.get_yticklabels(), ["#27ae60", "#e74c3c"]):
            tick.set_color(color)
        ax.set_ylim(-0.4, 1.4)
        ax.set_xlim(t_s, t_e)
        ax.set_xlabel("Time (s from start)", fontsize=9)
        ax.set_title(f"{name}\n[{t_s:.1f} s – {t_e:.1f} s]",
                     fontsize=11, fontweight="bold", color=col)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9,
                  markerscale=4, handletextpad=0.4)
        ax.grid(axis="x", ls="--", lw=0.4, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_edgecolor(col)
        ax.spines["bottom"].set_edgecolor(col)
        ax.spines["left"].set_linewidth(1.6)
        ax.spines["bottom"].set_linewidth(1.6)
        ax.set_facecolor("#fafafa")

        if n_a_w == 0:
            ax.text(0.5, 0.5, "No attacks in this window",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#aaaaaa", style="italic")

    fig.suptitle("Label vs Time — Beginning · Middle · End",
                 fontsize=14, fontweight="bold", y=1.01)

    out_path = os.path.join(output_dir, "label_vs_time_windows.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(out_path)}")


def plot_label_vs_time(df, output_dir):

    t0          = df["timestamp"].min()
    total_time  = df["timestamp"].max() - t0
    normal_mask = df["flag"].astype(str).str.strip().isin(["R", "0", "Normal"])

    n_bins = 500
    bins   = np.linspace(0, total_time, n_bins + 1)

    normal_counts, _ = np.histogram(df.loc[normal_mask,  "timestamp"] - t0, bins=bins)
    attack_counts, _ = np.histogram(df.loc[~normal_mask, "timestamp"] - t0, bins=bins)

    def _norm(arr):
        m = arr.max()
        return arr / m if m > 0 else arr.astype(float)

    labels  = ["Normal", "Attack"]
    counts  = [normal_counts, attack_counts]
    cmaps   = ["Blues", "Reds"]
    totals  = [int(normal_mask.sum()), int((~normal_mask).sum())]

    fig, axes = plt.subplots(2, 1, figsize=(16, 4), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor("#f8f9fa")

    for ax, label, cnt, cmap_name, total in zip(axes, labels, counts, cmaps, totals):
        ax.set_facecolor("#f8f9fa")
        mesh = ax.pcolormesh(
            bins, [0, 1], _norm(cnt).reshape(1, -1),
            cmap=cmap_name, vmin=0, vmax=1, shading="flat",
        )
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["0", f"{cnt.max()//2:,.0f}", f"{cnt.max():,.0f}"])
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("msg / bin", fontsize=7)

        ax.set_yticks([0.5])
        ax.set_yticklabels([f"{label}\n({total:,})"], fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    axes[-1].tick_params(axis="x", labelsize=10)
    fig.suptitle("Label vs Timestamp — Message Density",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_vs_time.png"),
                dpi=150, bbox_inches="tight")
    plt.close()