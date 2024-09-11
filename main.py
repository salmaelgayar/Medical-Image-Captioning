def function generate_caption(image, tokenizer, max_length, cnn_model,transformer_model):
  # Preprocess the image
 preprocessed_image = preprocess_image(image)
  # Extract features from the image using the CNN model
  image_features = cnn_model(preprocessed_image)
  # Initialize caption with start token
  start_token = tokenizer.word_index['<start>']
  current_caption = [start_token]
  # Loop for maximum caption length
  for  in range(max_length):
    # Convert caption to sequence of integer indices
    caption_ids = tokenizer.texts_to_sequences([current_caption])[0]
    # Encode caption and image features using Transformer encoder
    encoded_caption = transformer_model.encoder(caption_ids, image_features)
    # Predict next word probability distribution using Transformer decoder
    next_word_probs = transformer_model.decoder(encoded_caption, current_caption)
    # Sample next word from the probability distribution
    predicted_index = sample_next_word(next_word_probs)
    # Append the predicted word to the caption
    current_caption.append(predicted_index)
    # Check for end token
    if predicted_index==tokenizer.word_index['<end>']:
      break
  # Convert caption indices back to words
  generated_caption = tokenizer.sequences_to_texts([current_caption])[0]
  # Return the generated caption
  return generated_caption
