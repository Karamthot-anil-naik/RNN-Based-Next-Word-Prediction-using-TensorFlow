import tensorflow as tf

TEXT = (
   "i like machine learning",
   "machine learning is useful",
   "i like natural language processing",
   "natural language processing is interesting",
   "machine learning uses data",
   "data helps machine learning",
   "I like machine learning",
   "I like deep learning",
   "I love machine learning",
   "machine learning is interesting",
   "deep learning is powerful"
)
TRAINING_TEXT = " ".join(TEXT)

SEQUENCE_LENGTH = 4

vectorizer = None
model = None
vocabulary = None


def _build_vectorizer():
    vectorizer = tf.keras.layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        output_mode="int",
    )
    vectorizer.adapt(tf.constant([TRAINING_TEXT]))
    return vectorizer


def _prepare_training_data(vectorizer_instance):
    tokens = vectorizer_instance(tf.constant([TRAINING_TEXT]))[0]
    tokens = tf.cast(tokens, tf.int32)

    inputs = []
    targets = []
    for index in range(SEQUENCE_LENGTH, tokens.shape[0]):
        inputs.append(tokens[index - SEQUENCE_LENGTH:index])
        targets.append(tokens[index])

    if not inputs:
        raise ValueError("Training data is empty. Please provide more text.")

    return tf.stack(inputs), tf.stack(targets)


def _build_model(vocab_size):
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.int32),
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32),
            tf.keras.layers.SimpleRNN(64),
            tf.keras.layers.Dense(vocab_size, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def ensure_model():
    global vectorizer, model, vocabulary

    if model is not None:
        return model

    vectorizer = _build_vectorizer()
    vocabulary = vectorizer.get_vocabulary()
    x_train, y_train = _prepare_training_data(vectorizer)
    model = _build_model(len(vocabulary))
    model.fit(x_train, y_train, epochs=300, verbose=0)
    return model


def predict_next_word(seed_text):
    ensure_model()

    seed_tokens = vectorizer(tf.constant([seed_text]))[0]
    seed_tokens = tf.cast(seed_tokens, tf.int32)
    seed_tokens = seed_tokens[-SEQUENCE_LENGTH:]

    padding_size = max(0, SEQUENCE_LENGTH - int(seed_tokens.shape[0]))
    if padding_size:
        seed_tokens = tf.pad(seed_tokens, [[padding_size, 0]], constant_values=0)

    input_tensor = tf.expand_dims(seed_tokens, axis=0)
    prediction = model(input_tensor, training=False)
    predicted_id = int(tf.argmax(prediction[0]).numpy())
    return vocabulary[predicted_id]


def main():
    import streamlit as st

    global vectorizer, model, vocabulary

    @st.cache_resource
    def get_model_components():
        ensure_model()
        return vectorizer, model, vocabulary

    st.set_page_config(page_title="Next Word Predictor", page_icon="📝")
    st.title("Next Word Predictor")
    st.write("Enter a phrase and use a TensorFlow SimpleRNN model to predict the next word.")
    st.subheader("Training text")
    st.code("\n".join(TEXT), language=None)

    with st.form("prediction_form"):
        seed_text = st.text_input("Your phrase", value="i like machine")
        submitted = st.form_submit_button("Predict next word", type="primary")

    if not submitted:
        return None

    if not seed_text.strip():
        st.warning("Enter a phrase to get a prediction.")
        return None

    try:
        with st.spinner("Training the model and generating a prediction..."):
            vectorizer, model, vocabulary = get_model_components()
            next_word = predict_next_word(seed_text)
    except ValueError as error:
        st.error(str(error))
        return None

    st.subheader("Predicted Next Word")
    st.success(next_word)
    return next_word


if __name__ == "__main__":
    main()