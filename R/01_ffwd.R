# Clear Workspace
# ------------------------------------------------------------------------------
rm(list=ls())

# Load libraries
# ------------------------------------------------------------------------------
library("keras")
library("tidyverse")
library("ggseqlogo")
library("PepTools")

# Load data
# ------------------------------------------------------------------------------
pep_file = "https://raw.githubusercontent.com/leonjessen/keras_tensorflow_demo/master/data/ran_peps_netMHCpan40_predicted_A0201_reduced_cleaned_balanced.tsv"
pep_dat  = read_tsv(file = pep_file)

# Prepare Data for TensorFlow
# ------------------------------------------------------------------------------

# Set x/y test/train
x_train = pep_dat %>% filter(data_type == 'train') %>% pull(peptide)   %>% pep_encode
y_train = pep_dat %>% filter(data_type == 'train') %>% pull(label_num) %>% array
x_test  = pep_dat %>% filter(data_type == 'test')  %>% pull(peptide)   %>% pep_encode
y_test  = pep_dat %>% filter(data_type == 'test')  %>% pull(label_num) %>% array

# Reshape
x_train = array_reshape(x_train, c(nrow(x_train), 180))
x_test  = array_reshape(x_test,  c(nrow(x_test), 180))
y_train = to_categorical(y_train, y_train %>% table %>% length)
y_test  = to_categorical(y_test,  y_test  %>% table %>% length)

# Define the model
# ------------------------------------------------------------------------------
set.seed(922019)
model = keras_model_sequential() 
model %>% 
  layer_dense(units  = 180, activation = 'relu', input_shape = 180) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units  = 90, activation  = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units  = 3, activation   = 'softmax')

model %>% compile(
  loss      = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics   = c('accuracy')
)

# Run model
# ------------------------------------------------------------------------------
history = model %>% fit(
  x_train, y_train, 
  epochs = 150, batch_size = 50, validation_split = 0.2)

# Training
# ------------------------------------------------------------------------------
training_dat = tibble(epoch = rep(1:history$params$epochs,2),
                      value = c(history$metrics$acc,history$metrics$val_acc),
                      dtype = c(rep('acc',history$params$epochs),
                                rep('val_acc',history$params$epochs)) %>% factor)

# Performance
# ------------------------------------------------------------------------------
perf    = model %>% evaluate(x_test, y_test)
acc     = perf$acc %>% round(3) * 100
y_pred  = model %>% predict_classes(x_test)
y_real  = y_test %>% apply(1,function(x){ return( which(x==1) - 1) })
results = tibble(y_real  = y_real %>% factor,
                 y_pred  = y_pred %>% factor,
                 Correct = ifelse(y_real == y_pred,"yes","no") %>% factor)

# Visualise
# ------------------------------------------------------------------------------
training_dat %>%
  ggplot(aes(x = epoch, y = value, colour = dtype)) +
  geom_line() +
  geom_hline(aes(yintercept = perf$acc, linetype = 'Final performance')) +
  ggtitle(label = "Neural Network Training") +
  labs(x = 'Epoch number', y = 'Accuracy', colour = 'Data type') +
  scale_linetype_manual(name = 'Lines', values = 'dashed') +
  scale_color_manual(labels = c('Traning', 'Test'),
                     values = c('tomato','cornflowerblue')) +
  theme_bw()
ggsave(filename = 'plots/ffwd_01_test_training_over_epochs.png',
       width = 10, height = 6)

results %>%
  ggplot(aes(x = y_pred, y = y_real, colour = Correct)) +
  geom_point() +
  xlab("Measured (Real class, as predicted by netMHCpan-4.0)") +
  ylab("Predicted (Class assigned by Keras/TensorFlow deep FFWD ANN)") +
  ggtitle(label    = "Performance on 10% unseen data - FFWD Neural Network",
          subtitle = paste0("Accuracy = ", acc,"%")) +
  scale_color_manual(labels = c('No', 'Yes'),
                     values = c('tomato','cornflowerblue')) +
  geom_jitter() +
  theme_bw()
ggsave(filename = 'plots/ffwd_02_results_3_by_3_confusion_matrix_like.png',
       width = 10, height = 6)
