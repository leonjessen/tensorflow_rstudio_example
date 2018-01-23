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

# Settings
epochs           = 30
batch_size       = 50
validation_split = 0.2
num_classes      = 3
img_rows         = 9 
img_cols         = 20
img_channels     = 1
input_shape      = c(img_rows, img_cols, img_channels)


# Set x/y test/train
x_train = pep_dat %>% filter(data_type == 'train') %>% pull(peptide)   %>% pep_encode
y_train = pep_dat %>% filter(data_type == 'train') %>% pull(label_num) %>% array
x_test  = pep_dat %>% filter(data_type == 'test')  %>% pull(peptide)   %>% pep_encode
y_test  = pep_dat %>% filter(data_type == 'test')  %>% pull(label_num) %>% array

# Reshape
x_train = array_reshape(x_train, c(nrow(x_train), input_shape))
x_test  = array_reshape(x_test, c(nrow(x_test), input_shape))
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)

# Define the model
# ------------------------------------------------------------------------------

# Initialize sequential model
model = keras_model_sequential()

# Build architecture
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

model %>% compile(
  loss      = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
  metrics   = c('accuracy')
)

# Run model
# ------------------------------------------------------------------------------
history = model %>% fit(
  x_train, y_train, 
  epochs           = epochs,
  batch_size       = batch_size,
  validation_split = validation_split)

# Visualise Training
# ------------------------------------------------------------------------------
plot_dat = tibble(epoch = rep(1:history$params$epochs,2),
                  value = c(history$metrics$acc,history$metrics$val_acc),
                  dtype = c(rep('acc',history$params$epochs),
                            rep('val_acc',history$params$epochs)) %>% factor)
plot_dat %>%
  ggplot(aes(x = epoch, y = value, colour = dtype)) +
  geom_line() +
  theme_bw()

# Performance
# ------------------------------------------------------------------------------
perf    = model %>% evaluate(x_test, y_test)
acc     = perf$acc %>% round(3) * 100
y_pred  = model %>% predict_classes(x_test)
y_real  = y_test %>% apply(1,function(x){ return( which(x==1) - 1) })
results = tibble(y_real = y_real, y_pred = y_pred,
                 Correct = ifelse(y_real == y_pred,"yes","no") %>% factor)
results %>%
  ggplot(aes(x = y_pred, y = y_real, colour = Correct)) +
  geom_point() +
  xlab("Measured (Real class, as predicted by netMHCpan-4.0)") +
  ylab("Predicted (Class assigned by Keras/TensorFlow deep FFWD ANN)") +
  ggtitle(label    = "Performance on 10% unseen data",
          subtitle = paste0("Accuracy = ", acc,"%")) +
  scale_x_continuous(breaks = c(0,1,2), minor_breaks = NULL) +
  scale_y_continuous(breaks = c(0,1,2), minor_breaks = NULL) +
  geom_jitter() +
  theme_bw()

