# Clear Workspace
# ------------------------------------------------------------------------------
rm(list=ls())

# Load libraries
# ------------------------------------------------------------------------------
library("tidyverse")
library("PepTools")
library('randomForest')

# Load data
# ------------------------------------------------------------------------------
pep_file = paste0(
  "https://raw.githubusercontent.com/leonjessen/",
  "keras_tensorflow_demo/master/data/",
  "ran_peps_netMHCpan40_predicted_A0201_reduced_cleaned_balanced.tsv")
pep_dat  = read_tsv(file = pep_file)

# Training settings
# ------------------------------------------------------------------------------
n_tree      = 100
plot_width  = 10
plot_height = 6

# Prepare Data for Random Forest
# ------------------------------------------------------------------------------

# Setup training data
target  = 'train'
x_train = pep_dat %>% filter(data_type==target) %>% pull(peptide) %>%
  pep_encode_mat %>% select(-peptide)
y_train = pep_dat %>% filter(data_type==target) %>% pull(label_num) %>% factor

# Setup test data
target = 'test'
x_test = pep_dat %>% filter(data_type==target) %>% pull(peptide) %>%
  pep_encode_mat %>% select(-peptide)
y_test = pep_dat %>% filter(data_type==target) %>% pull(label_num) %>% factor

# Run model
# ------------------------------------------------------------------------------
rf_classifier = randomForest(x = x_train, y = y_train, ntree = n_tree)

# Performance
# ------------------------------------------------------------------------------
y_pred    = predict(rf_classifier, x_test)
n_correct = table(observed = y_test, predicted = y_pred) %>% diag %>% sum
acc       = (n_correct / length(y_test)) %>% round(3) * 100
results   = tibble(y_real  = y_test,
                   y_pred  = y_pred,
                   Correct = ifelse(y_real == y_pred,"yes","no") %>% factor)

# Visualise
# ------------------------------------------------------------------------------
# Perfomance plot
title = "Performance on 10% unseen data - Random Forest"
xlab  = "Measured (Real class, as predicted by netMHCpan-4.0)"
ylab  = "Predicted (Class assigned by random forest)"
f_out = "plots/03_rf_01_results_3_by_3_confusion_matrix.png"
results %>%
  ggplot(aes(x = y_pred, y = y_real, colour = Correct)) +
  geom_point() +
  xlab(xlab) +
  ylab(ylab) +
  ggtitle(label = title, subtitle = paste0("Accuracy = ", acc,"%")) +
  scale_color_manual(labels = c('No', 'Yes'),
                     values = c('tomato','cornflowerblue')) +
  geom_jitter() +
  theme_bw()
ggsave(filename = f_out, width = plot_width, height = plot_height)
