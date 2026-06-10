import matplotlib.pyplot as plt

def plot_roc_prc_curves(roc_dict: dict[list], prc_dict: dict[list], title: str):

    """
    Plots the ROC and PRC curves for each fold of a cross-validation process.
    """

    recall, precision = prc_dict["recall"], prc_dict["precision"]
    fpr, tpr = roc_dict["fpr"], roc_dict["tpr"]
    n_folds = len(recall)

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 5))
    
    for i in range(n_folds):
        #PRC
        axes[0].plot(recall[i], precision[i], label = f"Fold {i+1}", marker = "v", 
                     mfc = "none", alpha = 0.95)
        #ROC
        axes[1].plot(fpr[i], tpr[i], label = f"Fold {i+1}", marker = "v", mfc = "none", alpha = 0.95)

    axes[0].set_title(f"Curva PRC - {title}")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")

    axes[1].set_title(f"Curva ROC - {title}")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    for i in range(len(axes)):
        axes[i].legend()
        axes[i].grid(linestyle = "dashdot", alpha = 0.5)