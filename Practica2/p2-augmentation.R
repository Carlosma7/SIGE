## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2020-2021
## Pablo Alfaro Goicoechea
## Carlos Morales Aguilera
## Modelo propio de red neuronal convolutiva con aumento de datos (Data augmentation)
## -------------------------------------------------------------------------------------

# Carga de bibliotecas
library(tidyverse)
library(tensorflow)
library(keras)
library(caret)
library(scales)

# Mostrar una imagen del conjunto de datos
img_sample <- image_load(path = './data/images/mini50_twoClasses/test/1/7e1cdy.jpg', target_size = c(150, 150))
img_sample_array <- array_reshape(image_to_array(img_sample), c(1, 150, 150, 3))
plot(as.raster(img_sample_array[1,,,] / 255))

# Carga de datos
dataset_dir           <- './data/images/medium10000_twoClasses/'
train_images_dir      <- paste0(dataset_dir, 'train')
val_images_dir        <- paste0(dataset_dir, 'val')
test_images_dir       <- paste0(dataset_dir, 'test')

# Generadores de imágenes (reescalado)
train_images_generator <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
val_images_generator   <- image_data_generator(rescale = 1/255)
test_images_generator  <- image_data_generator(rescale = 1/255)

# Definición de flujos de imágenes con generadores
train_generator_flow <- flow_images_from_directory(
  directory = train_images_dir,
  generator = train_images_generator,
  class_mode = 'categorical',
  batch_size = 128,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)

validation_generator_flow <- flow_images_from_directory(
  directory = val_images_dir,
  generator = val_images_generator,
  class_mode = 'categorical',
  batch_size = 128,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)

test_generator_flow <- flow_images_from_directory(
  directory = test_images_dir,
  generator = test_images_generator,
  class_mode = 'categorical',
  batch_size = 128,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)

# Definición de la red neuronal propia
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu", input_shape = c(64, 64, 3)) %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>% 
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_batch_normalization() %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 2, activation = "softmax")

# Compilación del modelo
# Entropía cruzada categórica y optimizador adam con ratio de aprendizaje 0.005
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr=0.005),
  metrics = c('accuracy')
)

# Entrenamiento del modelo
start_time <- Sys.time()
history <- model %>% 
  fit_generator(
    generator = train_generator_flow, 
    validation_data = validation_generator_flow,
    steps_per_epoch = 10,
    epochs = 50
  )

plot(history)

# Evaluación del modelo
metrics <- model %>% 
  evaluate_generator(test_generator_flow, steps = 5)

end_time <- Sys.time()

message("  loss: ", metrics[1])
message("  accuracy: ", metrics[2])
message("  time: ", end_time - start_time)

# Evaluación mediante matriz de confusión
predictions <- predict_generator(model, test_generator_flow, steps = 4)

y_true <- test_generator_flow$classes
y_pred <- ifelse(predictions[,1] > 0.5, 1, 0)

cm <- confusionMatrix(as.factor(y_true), as.factor(y_pred))
cm_prop <- prop.table(cm$table)
plot(cm$table)

cm_tibble <- as_tibble(cm$table)
ggplot(data = cm_tibble) + 
  geom_tile(aes(x=Reference, y=Prediction, fill=n), colour = "white") +
  geom_text(aes(x=Reference, y=Prediction, label=n), colour = "white") +
  scale_fill_continuous(trans = 'reverse')