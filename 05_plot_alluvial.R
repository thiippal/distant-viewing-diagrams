library('ggplot2')
library('ggalluvial')

df = read.csv('data/ai2d_to_ai2d-rst_category_mapping.csv')

ggplot(as.data.frame(df), aes(y=Number, axis1=AI2D.RST, axis2=AI2D)) +
    geom_alluvium(aes(fill = AI2D.RST), width = 0) +
    geom_stratum(width = 1/12, fill= "grey", color = "white") +
geom_label(stat = "stratum", label.size=0, alpha=0.4, size=3, aes(label = after_stat(stratum)), position="identity") +
    scale_x_discrete(limits = c("AI2D-RST", "AI2D"), expand = c(.025, .085)) +
    scale_fill_brewer(type = "qual", palette = "Set3") +
    theme(legend.position = "none")
