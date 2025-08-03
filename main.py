from vibe import VIBE

if __name__ == "__main__":
    model_orchestrator = VIBE('bio', 'gender', 'biased_gender_data_synthetic.csv', hidden_layer_sizes=[500, 700], enable_debiasing=False, epochs=100, learning_rate=0.001)
    model_orchestrator.run()
    debiased_input = model_orchestrator.fetch_debiased_embeddings()