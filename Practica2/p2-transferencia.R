## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2020-2021
## Pablo Alfaro Goicoechea
## Carlos Morales Aguilera
## Técnica de transferencia de aprendizaje de modelos de redes neuronales
## -------------------------------------------------------------------------------------

# Carga de bibliotecas
library(keras)

# Carga de datos
dataset_dir           <- './data/images/medium10000_twoClasses/'
train_images_dir      <- paste0(dataset_dir, 'train')
val_images_dir        <- paste0(dataset_dir, 'val')
test_images_dir       <- paste0(dataset_dir, 'test')

# Generadores de imágenes (reescalado)
train_images_generator <- image_data_generator(rescale = 1/255)
val_images_generator   <- image_data_generator(rescale = 1/255)
test_images_generator  <- image_data_generator(rescale = 1/255)

# Definición de flujos de imágenes con generadores
train_generator_flow <- flow_images_from_directory(
  directory = train_images_dir,
  generator = train_images_generator,
  class_mode = 'binary',
  batch_size = 128,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)

validation_generator_flow <- flow_images_from_directory(
  directory = val_images_dir,
  generator = val_images_generator,
  class_mode = 'binary',
  batch_size = 128,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)

test_generator_flow <- flow_images_from_directory(
  directory = test_images_dir,
  generator = test_images_generator,
  class_mode = 'binary',
  batch_size = 128,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)

# Extracción de características red VGG16, con ImageNet
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(64, 64, 3)
)

# Congelar las capas convolutivas
freeze_weights(conv_base)

# Definición de la red neuronal
model <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compilación del modelo
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# Entrenamiento del modelo
start_time <- Sys.time()
history <- model %>% 
  fit_generator(
    train_generator_flow,
    steps_per_epoch = 10,
    epochs = 50,
    validation_data = validation_generator_flow
  )

plot(history)

# Evaluación del modelo
metrics <- model %>% evaluate_generator(test_generator_flow, steps = 5)

end_time <- Sys.time()

message("  loss: ", metrics[1])
message("  accuracy: ", metrics[2])
message("  time: ", end_time - start_time)

## --------------------------
## Fine tuning
## --------------------------

# 4. Descongelar una parte de la la capa base
unfreeze_weights(conv_base, from = "block2_conv1")
unfreeze_weights(conv_base, from = "block3_conv1")

# Compilación del modelo
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy")
)

# Entrenamiento del modelo
start_time <- Sys.time()
history <- model %>% 
  fit_generator(
    train_generator_flow,
    steps_per_epoch = 10,
    epochs = 50,
    validation_data = validation_generator_flow
  )

# Evaluación modelo
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
