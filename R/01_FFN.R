# Clear Workspace
# ------------------------------------------------------------------------------
rm(list=ls())

# Load libraries
# ------------------------------------------------------------------------------
library("keras")
library("tidyverse")
library("PepTools")

# Load data
# ------------------------------------------------------------------------------
pep_file = paste0(
  "https://raw.githubusercontent.com/leonjessen/",
  "keras_tensorflow_demo/master/data/",
  "ran_peps_netMHCpan40_predicted_A0201_reduced_cleaned_balanced.tsv")
pep_dat  = read_tsv(file = pep_file)

# Training settings
# ------------------------------------------------------------------------------
epochs           = 100
batch_size       = 50
validation_split = 0.2
num_classes      = 3
img_rows         = 9 
img_cols         = 20
img_channels     = 1
input_shape      = img_rows * img_cols
plot_width       = 10
plot_height      = 6

# Prepare Data for TensorFlow
# ------------------------------------------------------------------------------

# Setup training data
target  = 'train'
x_train = pep_dat %>% filter(data_type==target) %>% pull(peptide) %>% pep_encode
y_train = pep_dat %>% filter(data_type==target) %>% pull(label_num) %>% array

# Setup test data
target = 'test'
x_test = pep_dat %>% filter(data_type==target) %>% pull(peptide) %>% pep_encode
y_test = pep_dat %>% filter(data_type==target) %>% pull(label_num) %>% array

# Reshape
x_train = array_reshape(x_train, c(nrow(x_train), input_shape))
x_test  = array_reshape(x_test,  c(nrow(x_test),  input_shape))
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test,  num_classes)

# Define the model
# ------------------------------------------------------------------------------

# Initialize sequential model
model = keras_model_sequential() 

# Build architecture
model %>% 
  layer_dense(units  = 180, activation = 'relu', input_shape = 180) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units  = 90, activation  = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units  = 3, activation   = 'softmax')

# Compile model
model %>% compile(
  loss      = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics   = 'accuracy'
)

# Run model
# ------------------------------------------------------------------------------
history = model %>% fit(
  x_train, y_train, 
  epochs           = epochs,
  batch_size       = batch_size,
  validation_split = validation_split)

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
# Training plot
title = 'Neural Network Training - Feed Forward Neural Network'
xlab  = 'Epoch number'
ylab  = 'Accuracy'
f_out = 'plots/01_ffn_01_test_training_over_epochs.png'
training_dat %>%
  ggplot(aes(x = epoch, y = value, colour = dtype)) +
  geom_line() +
  geom_hline(aes(yintercept = perf$acc, linetype = 'Final performance')) +
  ggtitle(label = title) +
  labs(x = xlab, y = ylab, colour = 'Data type') +
  scale_linetype_manual(name = 'Lines', values = 'dashed') +
  scale_color_manual(labels = c('Traning', 'Test'),
                     values = c('tomato','cornflowerblue')) +
  theme_bw()
ggsave(filename = f_out, width = plot_width, height = plot_height)

# Perfomance plot
title = 'Performance on 10% unseen data - Feed Forward Neural Network'
xlab  = 'Measured (Real class, as predicted by netMHCpan-4.0)'
ylab  = 'Predicted (Class assigned by Keras/TensorFlow deep FFN)'
f_out = 'plots/01_ffn_02_results_3_by_3_confusion_matrix.png'
results %>%
  ggplot(aes(x = y_pred, y = y_real, colour = Correct)) +
  geom_point() +
  ggtitle(label = title, subtitle = paste0("Accuracy = ", acc,"%")) +
  xlab(xlab) +
  ylab(ylab) +
  scale_color_manual(labels = c('No', 'Yes'),
                     values = c('tomato','cornflowerblue')) +
  geom_jitter() +
  theme_bw()
ggsave(filename = f_out, width = plot_width, height = plot_height)

