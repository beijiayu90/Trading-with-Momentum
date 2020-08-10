from collections import OrderedDict
import pandas as pd
import numpy as np
import copy
import collections

from tests import generate_random_tickers, assert_output, project_test

@project_test
def test_resample_prices(fn):
    tickers = generate_random_tickers(5)
    dates = pd.DatetimeIndex(['2008-08-19', '2008-09-08', '2008-09-28', '2008-10-18', '2008-11-07', '2008-11-27'])
    resampled_dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'close_prices': pd.DataFrame(
            [
                [21.050810483942833, 17.013843810658827, 10.984503755486879, 11.248093428369392, 12.961712733997235],
                [15.63570258751384, 14.69054309070934, 11.353027688995159, 475.74195118202061, 11.959640427803022],
                [482.34539247360806, 35.202580592515041, 3516.5416782257166, 66.405314327318209, 13.503960481087077],
                [10.918933017418304, 17.9086438675435, 24.801265417692324, 12.488954191854916, 10.52435923388642],
                [10.675971965144655, 12.749401436636365, 11.805257579935713, 21.539039489843024, 19.99766036804861],
                [11.545495378369814, 23.981468434099405, 24.974763062186504, 36.031962102997689, 14.304332320024963]],
            dates, tickers),
        'freq': 'M'}
    fn_correct_outputs = OrderedDict([
        (
            'prices_resampled',
            pd.DataFrame(
                [
                        [21.05081048, 17.01384381, 10.98450376, 11.24809343, 12.96171273],
                        [482.34539247, 35.20258059, 3516.54167823, 66.40531433, 13.50396048],
                        [10.91893302, 17.90864387, 24.80126542, 12.48895419, 10.52435923],
                        [11.54549538, 23.98146843, 24.97476306, 36.03196210, 14.30433232]],
                resampled_dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_compute_log_returns(fn):
    tickers = generate_random_tickers(5)
    dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'prices': pd.DataFrame(
            [
                    [21.05081048, 17.01384381, 10.98450376, 11.24809343, 12.96171273],
                    [482.34539247, 35.20258059, 3516.54167823, 66.40531433, 13.50396048],
                    [10.91893302, 17.90864387, 24.80126542, 12.48895419, 10.52435923],
                    [11.54549538, 23.98146843, 24.97476306, 36.03196210, 14.30433232]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'log_returns',
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [3.13172138, 0.72709204, 5.76874778, 1.77557845, 0.04098317],
                    [-3.78816218, -0.67583590, -4.95433863, -1.67093250, -0.24929051],
                    [0.05579709, 0.29199789, 0.00697116, 1.05956179, 0.30686995]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_shift_returns(fn):
    tickers = generate_random_tickers(5)
    dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [3.13172138, 0.72709204, 5.76874778, 1.77557845, 0.04098317],
                [-3.78816218, -0.67583590, -4.95433863, -1.67093250, -0.24929051],
                [0.05579709, 0.29199789, 0.00697116, 1.05956179, 0.30686995]],
            dates, tickers),
        'shift_n': 1}
    fn_correct_outputs = OrderedDict([
        (
            'shifted_returns',
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [3.13172138, 0.72709204, 5.76874778, 1.77557845, 0.04098317],
                    [-3.78816218, -0.67583590, -4.95433863, -1.67093250, -0.24929051]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_get_top_n(fn):
    tickers = generate_random_tickers(5)
    dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'prev_returns': pd.DataFrame(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [3.13172138, 0.72709204, 5.76874778, 1.77557845, 0.04098317],
                [-3.78816218, -0.67583590, -4.95433863, -1.67093250, -0.24929051]],
            dates, tickers),
        'top_n': 3}
    fn_correct_outputs = OrderedDict([
        (
            'top_stocks',
            pd.DataFrame(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 1, 1, 0],
                    [0, 1, 0, 1, 1]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_portfolio_returns(fn):
    tickers = generate_random_tickers(5)
    dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'df_long': pd.DataFrame(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0],
                [0, 1, 0, 1, 1]],
            dates, tickers),
        'df_short': pd.DataFrame(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 1],
                [1, 1, 1, 0, 0]],
            dates, tickers),
        'lookahead_returns': pd.DataFrame(
            [
                [3.13172138, 0.72709204, 5.76874778, 1.77557845, 0.04098317],
                [-3.78816218, -0.67583590, -4.95433863, -1.67093250, -0.24929051],
                [0.05579709, 0.29199789, 0.00697116, 1.05956179, 0.30686995],
                [1.25459098, 6.87369275, 2.58265839, 6.92676837, 0.84632677]],
            dates, tickers),
        'n_stocks': 3}
    fn_correct_outputs = OrderedDict([
        (
            'portfolio_returns',
            pd.DataFrame(
                [
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [-0.00000000, -0.00000000, -0.00000000, -0.00000000, -0.00000000],
                    [0.01859903, -0.09733263, 0.00232372, 0.00000000, -0.10228998],
                    [-0.41819699, 0.00000000, -0.86088613, 2.30892279, 0.28210892]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_analyze_alpha(fn):
    dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'expected_portfolio_returns_by_date': pd.Series(
            [0.00000000, 0.00000000, 0.01859903, -0.41819699],
            dates)}
    fn_correct_outputs = OrderedDict([
        (
            't_value',
            -0.940764456618),
        (
            'p_value',
            0.208114098207)])

    assert_output(fn, fn_inputs, fn_correct_outputs)


pd.options.display.float_format = '{:.8f}'.format


def _generate_output_error_msg(fn_name, fn_inputs, fn_outputs, fn_expected_outputs):
    formatted_inputs = []
    formatted_outputs = []
    formatted_expected_outputs = []

    for input_name, input_value in fn_inputs.items():
        formatted_outputs.append('INPUT {}:\n{}\n'.format(
            input_name, str(input_value)))
    for output_name, output_value in fn_outputs.items():
        formatted_outputs.append('OUTPUT {}:\n{}\n'.format(
            output_name, str(output_value)))
    for expected_output_name, expected_output_value in fn_expected_outputs.items():
        formatted_expected_outputs.append('EXPECTED OUTPUT FOR {}:\n{}\n'.format(
            expected_output_name, str(expected_output_value)))

    return 'Wrong value for {}.\n' \
           '{}\n' \
           '{}\n' \
           '{}' \
        .format(
            fn_name,
            '\n'.join(formatted_inputs),
            '\n'.join(formatted_outputs),
            '\n'.join(formatted_expected_outputs))


def _is_equal(x, y):
    is_equal = False

    if isinstance(x, pd.DataFrame) or isinstance(y, pd.Series):
        is_equal = x.equals(y)
    elif isinstance(x, np.ndarray):
        is_equal = np.array_equal(x, y)
    elif isinstance(x, list):
        if len(x) == len(y):
            for x_item, y_item in zip(x, y):
                if not _is_equal(x_item, y_item):
                    break
            else:
                is_equal = True
    else:
        is_equal = x == y

    return is_equal


def project_test(func):
    def func_wrapper(*args):
        result = func(*args)
        print('Tests Passed')
        return result

    return func_wrapper


def generate_random_tickers(n_tickers=None):
    min_ticker_len = 3
    max_ticker_len = 5
    tickers = []

    if not n_tickers:
        n_tickers = np.random.randint(8, 14)

    ticker_symbol_random = np.random.randint(ord('A'), ord('Z')+1, (n_tickers, max_ticker_len))
    ticker_symbol_lengths = np.random.randint(min_ticker_len, max_ticker_len, n_tickers)
    for ticker_symbol_rand, ticker_symbol_length in zip(ticker_symbol_random, ticker_symbol_lengths):
        ticker_symbol = ''.join([chr(c_id) for c_id in ticker_symbol_rand[:ticker_symbol_length]])
        tickers.append(ticker_symbol)

    return tickers


def generate_random_dates(n_days=None):
    if not n_days:
        n_days = np.random.randint(14, 20)

    start_year = np.random.randint(1999, 2017)
    start_month = np.random.randint(1, 12)
    start_day = np.random.randint(1, 29)
    start_date = date(start_year, start_month, start_day)

    dates = []
    for i in range(n_days):
        dates.append(start_date + timedelta(days=i))

    return dates


def assert_structure(received_obj, expected_obj, obj_name):
    assert isinstance(received_obj, type(expected_obj)), \
        'Wrong type for output {}. Got {}, expected {}'.format(obj_name, type(received_obj), type(expected_obj))

    if hasattr(expected_obj, 'shape'):
        assert received_obj.shape == expected_obj.shape, \
            'Wrong shape for output {}. Got {}, expected {}'.format(obj_name, received_obj.shape, expected_obj.shape)
    elif hasattr(expected_obj, '__len__'):
        assert len(received_obj) == len(expected_obj), \
            'Wrong len for output {}. Got {}, expected {}'.format(obj_name, len(received_obj), len(expected_obj))

    if type(expected_obj) == pd.DataFrame:
        assert set(received_obj.columns) == set(expected_obj.columns), \
            'Incorrect columns for output {}\n' \
            'COLUMNS:          {}\n' \
            'EXPECTED COLUMNS: {}'.format(obj_name, sorted(received_obj.columns), sorted(expected_obj.columns))

        # This is to catch a case where __equal__ says it's equal between different types
        assert set([type(i) for i in received_obj.columns]) == set([type(i) for i in expected_obj.columns]), \
            'Incorrect types in columns for output {}\n' \
            'COLUMNS:          {}\n' \
            'EXPECTED COLUMNS: {}'.format(obj_name, sorted(received_obj.columns), sorted(expected_obj.columns))

        for column in expected_obj.columns:
            assert received_obj[column].dtype == expected_obj[column].dtype, \
                'Incorrect type for output {}, column {}\n' \
                'Type:          {}\n' \
                'EXPECTED Type: {}'.format(obj_name, column, received_obj[column].dtype, expected_obj[column].dtype)

    if type(expected_obj) in {pd.DataFrame, pd.Series}:
        assert set(received_obj.index) == set(expected_obj.index), \
            'Incorrect indices for output {}\n' \
            'INDICES:          {}\n' \
            'EXPECTED INDICES: {}'.format(obj_name, sorted(received_obj.index), sorted(expected_obj.index))

        # This is to catch a case where __equal__ says it's equal between different types
        assert set([type(i) for i in received_obj.index]) == set([type(i) for i in expected_obj.index]), \
            'Incorrect types in indices for output {}\n' \
            'INDICES:          {}\n' \
            'EXPECTED INDICES: {}'.format(obj_name, sorted(received_obj.index), sorted(expected_obj.index))


def does_data_match(obj_a, obj_b):
    if type(obj_a) == pd.DataFrame:
        # Sort Columns
        obj_b = obj_b.sort_index(1)
        obj_a = obj_a.sort_index(1)

    if type(obj_a) in {pd.DataFrame, pd.Series}:
        # Sort Indices
        obj_b = obj_b.sort_index()
        obj_a = obj_a.sort_index()
    try:
        data_is_close = np.isclose(obj_b, obj_a, equal_nan=True)
    except TypeError:
        data_is_close = obj_b == obj_a
    else:
        if isinstance(obj_a, collections.Iterable):
            data_is_close = data_is_close.all()

    return data_is_close


def assert_output(fn, fn_inputs, fn_expected_outputs, check_parameter_changes=True):
    assert type(fn_expected_outputs) == OrderedDict

    if check_parameter_changes:
        fn_inputs_passed_in = copy.deepcopy(fn_inputs)
    else:
        fn_inputs_passed_in = fn_inputs

    fn_raw_out = fn(**fn_inputs_passed_in)

    # Check if inputs have changed
    if check_parameter_changes:
        for input_name, input_value in fn_inputs.items():
            passed_in_unchanged = _is_equal(input_value, fn_inputs_passed_in[input_name])

            assert passed_in_unchanged, 'Input parameter "{}" has been modified inside the function. ' \
                                        'The function shouldn\'t modify the function parameters.'.format(input_name)

    fn_outputs = OrderedDict()
    if len(fn_expected_outputs) == 1:
        fn_outputs[list(fn_expected_outputs)[0]] = fn_raw_out
    elif len(fn_expected_outputs) > 1:
        assert type(fn_raw_out) == tuple,\
            'Expecting function to return tuple, got type {}'.format(type(fn_raw_out))
        assert len(fn_raw_out) == len(fn_expected_outputs),\
            'Expected {} outputs in tuple, only found {} outputs'.format(len(fn_expected_outputs), len(fn_raw_out))
        for key_i, output_key in enumerate(fn_expected_outputs.keys()):
            fn_outputs[output_key] = fn_raw_out[key_i]

    err_message = _generate_output_error_msg(
        fn.__name__,
        fn_inputs,
        fn_outputs,
        fn_expected_outputs)

    for fn_out, (out_name, expected_out) in zip(fn_outputs.values(), fn_expected_outputs.items()):
        assert_structure(fn_out, expected_out, out_name)
        correct_data = does_data_match(expected_out, fn_out)

        assert correct_data, err_message

