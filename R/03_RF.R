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
pep_file = "https://raw.githubusercontent.com/leonjessen/keras_tensorflow_demo/master/data/ran_peps_netMHCpan40_predicted_A0201_reduced_cleaned_balanced.tsv"
pep_dat  = read_tsv(file = pep_file)

# Prepare Data for Random Forest
# ------------------------------------------------------------------------------
pep_enc = pep_dat %>% pull(peptide) %>% pep_encode_mat

train = pep_dat %>% filter(data_type == 'train') %>%
  select(peptide, label_num) %>% left_join(pep_enc, by='peptide') %>%
  select(-peptide)
train_x = train %>% select(-label_num)
train_y = train %>% pull(label_num) %>% factor

test = pep_dat %>% filter(data_type == 'test') %>%
  select(peptide, label_num) %>% left_join(pep_enc, by='peptide') %>%
  select(-peptide)
test_x = test %>% select(-label_num)
test_y = test %>% pull(label_num) %>% factor

# Training
# ------------------------------------------------------------------------------
rf_classifier = randomForest(x = train_x, y = train_y, ntree=150)

# Performance
# ------------------------------------------------------------------------------
y_pred = predict(rf_classifier,test_x)
table(observed = test_y, predicted = y_pred)
n_correct = table(observed = test_y, predicted = y_pred) %>%
  diag %>% sum
acc = (n_correct / length(test_y)) %>% round(3) * 100
results = tibble(y_real  = test_y,
                 y_pred  = y_pred,
                 Correct = ifelse(y_real == y_pred,"yes","no") %>% factor)

# Visualise
# ------------------------------------------------------------------------------
results %>%
  ggplot(aes(x = y_pred, y = y_real, colour = Correct)) +
  geom_point() +
  xlab("Measured (Real class, as predicted by netMHCpan-4.0)") +
  ylab("Predicted (Class assigned by random forest)") +
  ggtitle(label    = "Performance on 10% unseen data - Random Forest",
          subtitle = paste0("Accuracy = ", acc,"%")) +
  scale_color_manual(labels = c('No', 'Yes'),
                     values = c('tomato','cornflowerblue')) +
  geom_jitter() +
  theme_bw()
ggsave(filename = 'plots/rf_01_results_3_by_3_confusion_matrix_like.png',
       width = 10, height = 6)
