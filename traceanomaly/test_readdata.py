from . import readdata
import numpy as np


class TestClass:
    def test_read_raw(self):
        input_file = 'vae/case/case_raw1'
        y_flows = ['flow1', 'flow2', 'flow3']
        y_vecs = np.array([[200, 300, 0, 100], [0, 200, 300, 400], [500, 0, 0, 200]])
        y_vc = [0, 1, 2, 3]
        flows, vecs, vc = readdata.read_raw_vector(input_file)

        assert flows == y_flows
        assert (y_vecs == vecs).all() == True
        assert vc == y_vc

        input_file = 'vae/case/case_raw2'
        y_flows = ['flow4', 'flow5', 'flow6']
        y_vecs = np.array([[200, 0, 100], [0, 300, 0], [500, 0, 200]])
        y_vc = [0, 2, 3]
        flows, vecs, vc = readdata.read_raw_vector(input_file)

        assert flows == y_flows
        assert (y_vecs == vecs).all() == True
        assert vc == y_vc

        input_file = 'vae/case/case_raw3'
        y_flows = ['flow7', 'flow8', 'flow9', 'flow0']
        y_vecs = np.array([[200, 0], [0, 300], [500, 0], [23, 67]])
        y_vc = [0, 2]
        flows, vecs, vc = readdata.read_raw_vector(input_file)

        assert flows == y_flows
        assert (y_vecs == vecs).all() == True
        assert vc == y_vc

        input_file = 'vae/case/case_raw1'
        y_flows = ['flow1', 'flow2', 'flow3']
        y_vecs = np.array(
            [[300, 0, 100], [200, 300, 400], [0, 0, 200]])
        y_vc = [1, 2, 3]
        flows, vecs, vc = readdata.read_raw_vector(input_file, [1, 2, 3])

        assert flows == y_flows
        assert (y_vecs == vecs).all() == True
        assert vc == y_vc

    def test_get_mean_std(self):
        x = np.array([[1, 2, 3, 10], [4, 5, 6, 67], [72, 8, 9, 10]])
        y, z = readdata.get_mean_std(x)
        y1 = [25.66667, 5, 6, 29]
        z1 = [32.7855, 2.44949, 2.44949, 26.87006]

        assert len(y) == len(y1)
        assert len(y) == x.shape[1]

        for i in range(0, len(y1)):
            assert np.abs(y[i] - y1[i]) < 1e-5
            assert np.abs(z[i] - z1[i]) < 1e-5

    def test_normalization(self):
        x = np.array([[1, 2, 3, 10], [4, 5, 6, 67], [72, 8, 9, 10]])
        m = np.array([25.66667, 5, 6, 29])
        s = np.array([32.7855, 2.44949, 2.44949, 26.87006])

        y = readdata.normalization(x, m, s)

        assert x.shape == y.shape

        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                assert np.abs(y[i, j] - (x[i, j] - m[j]) / s[j]) < 1e-6

        x = np.array([
            [1, 0, 3, 10],
            [4, 0, 6, 67],
            [72, 0, 9, 10]])
        m = np.array([25.66667, 0, 6, 29])
        s = np.array([32.7855, 0, 0, 26.87006])
        y = readdata.normalization(x, m, s)

        assert x.shape == y.shape

        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if s[j] == 0:
                    if x[i, j] == m[j]:
                        assert y[i, j] == 0
                    elif x[i, j] > m[j]:
                        assert y[i, j] == 3
                    else:
                        print(x[i, j], m[j], y[i, j])
                        assert y[i, j] == -3
                else:
                    assert np.abs(y[i, j] - (x[i, j] - m[j]) / s[j]) < 1e-6

    def test_combine(self):
        file1 = 'vae/case/case_raw1'
        file2 = 'vae/case/case_raw2'
        file3 = 'vae/case/case_raw3'

        y = np.array([0, 0, 0, 1, 1, 1, 1])
        f = ['flow4', 'flow5', 'flow6', 'flow7', 'flow8', 'flow9', 'flow0']
        _, vecs, vc = readdata.read_raw_vector(file1)
        m, s = readdata.get_mean_std(vecs)

        (x0, y0), (x1, y1), fs = readdata.get_data_vae(file1, file2, file3)
        assert fs == f
        assert (y1 == y).all() == True
        print(y0)
        assert (y0 == np.zeros(len(vecs))).all() == True

        assert x1[6, 0] == (23 - m[0])/s[0]
        assert x1[3, 1] == (0 - m[1])/s[1]

    def test_get_z_dim(self):
        x = 100
        assert readdata.get_z_dim(x) == 10

        x = 20
        assert readdata.get_z_dim(x) == 5

        x = 31
        assert readdata.get_z_dim(x) == 10

        x = 400
        assert readdata.get_z_dim(x) == 10

        x = 420
        assert readdata.get_z_dim(x) == 20

        x = 500
        assert readdata.get_z_dim(x) == 20