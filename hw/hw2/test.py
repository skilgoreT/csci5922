def test(w):
    test_data = load_data('test_data.txt')
    # same transform pipeline on the test data as the training data
    test_prepared = pipeline.fit_transform(test_data)
    df = pd.DataFrame(test_prepared)
    df.plot(subplots=True, layout=(2,3), figsize=(12, 8));
    # add a bias to the prepared test data
    test_with_bias = np.full((len(test_prepared),6),1.0)
    test_with_bias[:,1:] = test_prepared
    # shape is (nrow, 6)
    X = test_with_bias
    t = [z if z == 1 else -1 for z in test_data['Occupancy']]
    # all we care about here is E
    _, E = dwp(w, t, X)
    print(1-E)

test(w_trained)