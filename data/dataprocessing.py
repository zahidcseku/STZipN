from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from datetime import date, datetime
import os.path
from math import radians

# import dataconfig from configs
from configs import data_config as dc

class GBRDataset(Dataset):
    """This class will handle data preparation and data statistics related tasks.

    Ideally we will use the dataclass of pytorch to make use of the batching, concurrency etc.
    In Version 0.0.1: does not include pytorch data class format.

    This class will generate the following files:
        * `featurematrix`: a feature matrix of dim (ndays, nsites, fdim)
        * `targets`: a matrix of dim (ndays, nsites, 1) corresponding to the COTS counts at all the sites at day t+1
        * `sites`: a list of sites
        * An adjacency matrix Aij depending on the gat parameter.

    The following parameters are initialized in the constructor:

    :param poissonmodel: describes whether the features will be constructed for poissonmodel or not. If set to true
        the targets will not be normalized.
    :type poissonmodel: boolean defaults to `True`
    :param verbose: whether to print the stats on screen or not.
    :type verbose: boolean defaults to `False`
    :param fdim: indicates feature dimension.

        * If ``fdim=1``, only feature is cot counts.
        * If ``fdim=2``, two features are cot counts and number of days since last visit.
        * if ``fdim=3``, daily weather feature added to ncots.
    :type fdim: integer (1,2,3)
    :param normalize: controls whether to normalize the features or not.
    :type normalize: boolean, defaults to ``True``.
    :param savefeatures: whether to save the features or not.
    :type savefeatures: bool
    :param fileloc: location of the cull file. The file should be an excel file.
    :type fileloc: str
    """

    def __init__(self, verbose=True, **kwargs):
        """Constructor setting initial values of various parameters.
        """

        # setting up the logger
        print(f'\n{"**"*20}\nExecuting data processing......')
        print(f'Parameters: \n'
                    f'\tPoisson:{kwargs["poissonmodel"]}\n'       
                    f'\tverbose: {verbose}\n'
                    f'\tpoissonmodel: {kwargs["poissonmodel"]}\n'
                    f'\tfdim: {kwargs["fdim"]}\n'
                    f'\tnormalize: {kwargs["normalize"]}\n'
                    f'\tsavefeatures: {kwargs["savefeatures"]}\n'
                    f'\tdata file location: {kwargs["fileloc"]}')

        self.poissonmodel = kwargs["poissonmodel"]
        self.verbose = verbose
        self.fdim = kwargs["fdim"]
        self.normalize = kwargs["normalize"]
        self.savefeatures = kwargs["savefeatures"]

        fileloc = kwargs["fileloc"]

        print(f'\n{"**"*20}\nExecuting data processing......')
        print(f'Reading data file from: {fileloc}')
        # load the data from cullfile
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.df = pd.read_excel(fileloc, engine='openpyxl')
        print(f'Panda DataFrame shape:{self.df.shape}.')

        print(f"Getting the sites, site features and locations....")
        self.sites, self.static_features, self.locations = self.parse_sites()
        print(f"Got {self.get_nsites()} sites, {self.static_features.shape} site features "
                      f"and {len(self.locations)} locations.")

        print(f"Parsing the data....")
        self.data = self.parse_data()
        print(f"Got a list of lists. length of data is {len(self.data)}.")

        # compute the distance
        print(f"Computing the distance matrix....")
        self.D = self.compute_distance()
        print(f"Distance metrix D with shape {self.D.shape}.")

        print(f"Normalizing the static features....")
        self.Z = self.normalize_static_features()
        print(f"Shape of normalized static features: {len(self.Z), len(self.Z[0])}.")

        print(f"Building the feature matrix....")
        X = self.build_feature_matrix(self.normalize)
        print(f"Shape of the feature matrix: {X.shape}.")

        self.X = X[:-1]
        # X and self.X are different
        self.Y = X[1:, :, 0]

        if self.savefeatures:
            date = datetime.today().strftime("%Y%m%d")
            # save the feature and target matrix
            np.savez(f"{dc.data_output_dir}/feature_mat_dim_{self.fdim}_norm_{self.normalize}_{date}.npz",  X=self.X, Y=self.Y)

            # save the sites
            np.savez(f"{dc.data_output_dir}/sites_{date}.npz", sites=self.sites)

            # save distance mat
            np.savez(f"{dc.data_output_dir}/distmat_{date}.npz", dist=self.D)

            # save the static features
            np.savez(f"{dc.data_output_dir}/satic_features_{date}.npz", static_features=self.static_features)

            # save adjaceny mat - should be moved here later
            # np.savez(f"{dc.data_output_dir}/adjmat_{date}.npz", adj=self.A)

    def __getitem__(self, idx):
        """Given an index (row number) in the feature matrix, returns the corresponding features and targets

        :param idx: the index in the feature matrix.
        :type idx: int
        :return: a dictionary of feature vector and target corresponding to `idx`
        :rtype: dictonary indexed x (feature vector) and y (target vector)
        """
        x = self.X[idx]
        y = self.Y[idx]

        return {'x': x,
                'y': y}

    def __len__(self):
        """Returns the length of feature matrix i.e., the number of rows.
        :return: length of the feature matrix.
        :rtype: int
        """
        return len(self.X)

    def get_nsites(self):
        """Returns the numer of sites in the dataset i.e., the number of columns.

        :return: the number of sites in the dataset.
        :rtype: int
        """
        return len(self.sites)

    def get_ndays(self):
        """
        Returns the number of observation days in the dataset. Last day is only for testing therefore we subtract 1.

        :return: the number of observation days.
        :rtype: int
        """
        return len(self.data) - 1

    def parse_data(self):
        """Builds the spatical temporal culling data matrix.

        The culling file has the following columns:
            * Reef Number:
            * Site Number: {1, 2, 3, ...}
            * Latitude of Site 
            * Longitude of Site
            * Date
            * Number of COTS Removed: {29, 5, 13, 32, ...}
        This method process the raw dataframe ``df`` and `sites` initialized from the cull file in the constructor.
            ``self.df`` has the following columns:
            * Reef Number
            * Site Number
            * Latitude of Site
            * Longitude of Site
            * Date
            * Number of COTS Removed
        ``sites`` is a list in the format  [ ( reefid, siteid), .... ] sorted by latitudes.

        :param df: a pandas dataframe loaded from the cull file.
        :type df: pandas dataframe.
        :param sites: a list in the format  [ ( reefid, siteid), .... ] sorted by latitudes.
        :type sites: list of tuples.

        :return data: a list of lists. length of data is day1 to dayn. For each day, it records the number of
            observations as a pair (id, no of COTS)
            example:  [(337, 440.0), (351, 476.0), (356, 1091.0)] - shows on this day (the day is indexed by the list
                index), there are three observations,
                * 440 COTS at site 337 (represents a site which can be identified from sites list)
                * 476 COTS at site 351
                * 1091 COTS at site 356
        :rtype: list of lists.
        """

        '''a list of dates '''
        dates = list(set(self.df['Date'].values))
        #for row in self.df.itertuples():
        #    year, month, day = row[5].split('-')
        #    current = date(int(year), int(month), int(day))
        #    dates.append(current)
        dates.sort()

        earliest_date = dates[0]
        latest_date = dates[-1]

        #num_days = (latest_date - earliest_date).days + 1
        num_days = len(dates)

        data = [[] for i in range(num_days)]
        site_to_id = {_site: _idx for _idx, _site in enumerate(self.sites)}

        if self.verbose:
            print('earliest:', earliest_date)
            print('latest:', latest_date)

            gap = []
            for d1, d2 in zip(dates, dates[1:]):
                gap.append((d2 - d1))

            print('max gap without between two voyage: {} days'.format(max(gap)))
            print(f'{num_days} days of data')

        for row in self.df.itertuples():
            # print(row)
            _reef = row[1].strip()
            _site = int(row[2])

            #year, month, day = row[5].split('-')
            current = row[5]

            date_idx = current - earliest_date
            site_idx = site_to_id[(_reef, _site)]

            data[date_idx].append((site_idx, row[7]))

        #if self.verbose:
        #    self.print_statistics(data, self.sites)

        return data

    def parse_sites(self):
        """Parses the Culling data file and get sites, static_features and locations

        :param df: the culll dataframe
        :type df: pandas dataframe object

        :returns sites: a list in the format  [ ( reefid, siteid), .... ] sorted by latitude.           
        :rtype sites: a list of tuples.
        :returns static_feature: a list of lists of the format [[latitude, longitude], [], ..]                
        :rtype static_feature: a list of lists.
        :returns locations: a dictionary of locations (latitude, longitude) indexed by
            (reef_id, site_id):(latitude, longitude)
            
        :rtype locations: a dictionary object.
        """

        locations = {}

        for row in self.df.itertuples():           
            _reef = row[1]#.strip()
            #_site = int(row[2])
            _site = row[2]

            if (_reef, _site) in locations:
                x, y = locations[(_reef, _site)]
                '''checking inconsistencies: very close sites are inconsistent '''
                #distance = np.sqrt((x - row[3]) ** 2 + (y - row[4]) ** 2)
                #if distance >= 1e-5:
                #    print('inconsistancy:', _reef, _site, locations[(_reef, _site)])
                #    print('inconsistancy:', _reef, _site, row[3], row[4])
                #    exit(0)
            else:
                locations[(_reef, _site)] = (row[3], row[4])

        ''' a list in the format  [ ( reefid, siteid, latitude, longitude ), .... ] '''
        sites = []
        for s in locations:
            sites.append(s + locations[s])

        ''' sort by latitude (from south to north) '''
        sites.sort(key=lambda x: x[2])

        # static site features:
        # Reef ID, latitude, longtitude, ...
        static_feature = []
        for x in sites:
            static_feature.append(np.hstack([x[2], x[3]]))

        static_feature = np.array(static_feature, dtype=np.float32)

        '''sites is now a list of two things [reefid, siteid]'''
        sites = [(x[0], x[1]) for x in sites]

        if self.verbose:
            print('{} sites'.format(len(sites)))

        return sites, static_feature, locations


    def compute_distance(self):
        """Computes the distances between sites.

        This method computes the haversine distance between two sites.
            `Haversine distance <https://https://en.wikipedia.org/wiki/Haversine_formula/>`.

        :param sites: a list in the format  [ ( reefid, siteid), .... ] sorted by latitude.
        :type sites: list of tuples.
        :param locations: a dictionary of locations (latitude, longitude) indexed by (reef_id, site_id).
            Example: (reef_id, site_id):(latitude, longitude)
        :type locations: a dictinary.

        :return: a 2D matrix of size sites X sites where dist[i, j] is the distance beteween site i and site j
            (when ``gat=False``) or two matrices H and V.
        """
        r = 6371e3

        phi, lamb = [], []
        for _idx in self.sites:
            phi.append(radians(self.locations[_idx][0]))
            lamb.append(radians(self.locations[_idx][1]))
        phi, lamb = np.array(phi),  np.array(lamb)

        deltaphi = phi - phi[:, None]
        deltalambda = lamb - lamb[:, None]

        a = np.cos(phi) * np.cos(phi[:, None]) * (np.sin(0.5 * deltalambda) ** 2)

        a = a + np.sin(0.5 * deltaphi) ** 2
        dist = 2 * r * np.arcsin(np.sqrt(a))

        return dist

    def get_locations(self):
        """
        Returns the locations from the locations data structure.

        :return: a dictionary of locations (latitude, longitude) indexed by
            (reef_id, site_id):(latitude, longitude)
        :rtype: a dictionary object.
        """
        return self.locations

    def get_distmat(self):
        """Returns the distance matrix.

        :return: the distance matrix.
        :rtype: a 2D numpy matrix of distances.
        """
        return self.D

    def print_statistics(self, interval=14, outfile=None):
        """Reports some useful data statistics

        The followings are reported:
            * Average voyages per day
            * Maximum voyages per day
            * Number of sites
            * Number of observation days
            * Minimum and maximum number of COTS
            * Average number of COTS per observed day
            * Percentages of COTS less than 10, 100 and 1000
            * Repeat visit ration within an interval (or voyage)
            * Number of non observed days
            * Data sparsity

        :param interval: usually the voyage length. Required to compute few statistics e.g., Repeat visit ration
            within an interval
        :type interval: int defaults to 14 days
        :param outfile: name of the file to store the statistics if not None.
        :type outfile: str defaults to None
        """

        daily_voyage = np.array([len(_data_day) for _data_day in self.data])
        outstr = f"Date time: {datetime.now()}\n"
        outstr += f'Average {daily_voyage.mean():.1f} voyages per day\n'
        outstr += f'Maxmium {daily_voyage.max()} voyages per day\n'

        number_cots = []
        for _data_day in self.data:
            for site, ncots in _data_day:
                number_cots.append(ncots)

        number_cots = np.array(number_cots)

        outstr += f"Number of sites: {self.X.shape[1]}\n"
        outstr += f"Number of days-span: {self.X.shape[0]}\n"
        outstr += f'minimum #cots: {number_cots.min()}\n'
        outstr += f'maximum #cots: {number_cots.max()}\n'
        outstr += f'average #cots: {number_cots.mean():.1f}\n'

        percentage = (number_cots < 1000).sum() / number_cots.size
        outstr += f'#cots < 1000: {percentage:.3f}\n'

        percentage = (number_cots < 100).sum() / number_cots.size
        outstr += f'#cots < 100: {percentage:.3f}\n'

        percentage = (number_cots < 10).sum() / number_cots.size
        outstr += f'#cots < 10: {percentage:.3f}\n'

        count = []
        ratio = []
        for i in range(len(self.data)):
            _data = self.data[i:i + interval]
            if len(_data) == interval:

                visited_sites = []
                c = 0
                for _data_day in _data:
                    visited_sites += [r[1] for r in _data_day]
                    c += len(_data_day)

                if len(visited_sites) > 0:
                    ratio.append(1 - len(set(visited_sites)) / len(visited_sites))
                    count.append(c)

        mean_ratio = np.mean(ratio)

        min_count = min(count)
        mean_count = np.mean(count)
        outstr += f'every {interval} days:\n'
        outstr += f'    repeat visit ratio: {mean_ratio:.2f}\n'
        outstr += f'    mininum {min_count} culling voyages\n'
        outstr += f'    average {mean_count:.1f} culling voyages\n'

        empty_days = len([None for _d in self.data if len(_d) == 0])
        outstr += f'{empty_days} days do not have data\n'

        total_entries = sum([len(_d) for _d in self.data])
        sparsity = total_entries / (len(self.data) * len(self.sites))
        outstr += f'sparsity: {sparsity:.4f}'

        # print statistics
        print(outstr)

        # save file if outfile is not None
        if outfile is not None:
            fname = f'Data processing-{outfile}-{date.today()}.txt'
            if os.path.isfile(fname):
                # file already exists so append
                mode = "a"
                outstr = "\n\n" + outstr
            else:
                # file does not exist so create new
                mode = "w"
            f = open(fname, mode)
            f.write(outstr)
            f.close()

    def build_feature_matrix(self, normalize):
        """ Preprocess features

        Pay special attention to:
        - the method to treat missing values;
        - the normalization method

        This method builds a feature matrix from data and sites.
        :param data: a list of tuples. Length of data is day1 to dayn. For each day, it records the number of
            observations as a pair (id, no of COTS). Example: `[(337, 440.0), (351, 476.0), (356, 1091.0)]`
        :type data: a list
        :param sites: a list in the format  [ ( reefid, siteid), .... ] sorted by latitude of the sites.
        :return: a 3D array _X of shape (ndays, nsites, fdim) where  _X[i, j] represents the number of COTS
            observed (or NCOTS/1000 if normalize=True) observed at site j on day i. Or -1 if no COTS observed at
            site j on day i.
        :rtype: a 3D tensor
        Notes:
            - the COTS count is normalized to the range 0 to 1 for non poisson models. 1 means >=1000 COTS (very rare)
            - Missing observations are imputed with -1
        """

        if 0 < self.fdim <= 3:
            _X = np.full([len(self.data), len(self.sites), self.fdim], -1, dtype=np.float32)
        else:
            print("Unknown feature dimension when building the feature matrix!!!")
            exit(0)

        """this loop will construct the feature matrix"""
        for i, drow in enumerate(self.data):
            for site, ncots in drow:
                assert (ncots >= 0)

                if not self.poissonmodel:
                    ncots = min(ncots / 1000, 1)

                # the COTS count is normalized to the range 0 to 1
                # 1 means >=1000 COTS (very rare)
                if self.fdim == 1:
                    if normalize:
                        _X[i, site] = min(ncots / 1000, 1)
                    else:
                        _X[i, site] = ncots

                elif self.fdim == 2:
                    if normalize:
                        _X[i, site] = [min(ncots / 1000, 1), -1]
                    else:
                        _X[i, site] = [ncots, -1]

        '''Fill in the day last_visit'''
        if self.fdim == 2:
            """iterate through each site and update the day_since_last_visit feature
            tilda measures the days between visits. The value of tilda = len(X) before the first visit
            then it starts counting from 0,1,...
            """
            for site in range(self.get_nsites()):
                tilda, firstvisit = self.get_ndays(), False
                for t, ncots in enumerate(_X[:, site, 0]):
                    if ncots > -1:
                        firstvisit = True
                        tilda = 0
                    if firstvisit:
                        tilda += 1

                    _X[t, site] = [ncots, tilda-1]

        return _X

    def normalize_static_features(self):
        """Normalizes the coordinates of the sites.

        :param static_features: the coordinates of the sites. A list of lists of the format [[latitude, longitude],
            [], ..].
        :type static_features: list of lists
        :return: a list of lists with normalized locations.
        """
        _mean = self.static_features.mean(0)
        _std = self.static_features.std(0)

        return (self.static_features - _mean) / (_std + np.finfo(np.float32).eps)

    def compute_kernel_width(self, num_neighbours=5):
        # binary search kernel width
        _N = self.D.shape[0]

        _min_h = 0
        _max_h = 1e5
        _h = 0.5 * (_min_h + _max_h)

        np.seterr(divide='ignore')
        while np.abs(self.entropy(_h) - np.log(_N * num_neighbours)).any() > 1e-5:
            if self.entropy(_h) > np.log(_N * num_neighbours).any():
                _max_h = _h
                _h = 0.5 * (_min_h + _max_h)
            else:
                _min_h = _h
                _h *= 2

            if self.verbose:
                print('h={} ent={}'.format(_h, np.exp(self.entropy(self.D, _h)) / _N))

        # some verification of row-entropy
        # P = np.exp( -dist**2/h**2 )
        # np.fill_diagonal(P, 0)
        # Psum = P.sum( 1 )
        # P/=Psum[:,None]
        # print( np.mean( np.exp( (-P*np.log(P+1e-20)).sum(1) ) ) )

        return _h

    def entropy(self, h):
        '''
        entropy in nats of the nxn matrix (remove self-similarities)
        '''
        _P = np.exp(-self.D ** 2 / h ** 2)
        np.fill_diagonal(_P, 0)
        _Psum = _P.sum()
        _P /= _Psum
        _ent = (-_P * np.log(_P + np.finfo(np.float32).eps)).sum()
        return _ent

    def zero_obs_ratio(self):
        """
        Prints statistics about the sparsity in the feature mattix.
        :return:
        """
        total_cells = self.X.shape[0] * self.X.shape[1]
        unobserved = np.count_nonzero(self.X == -1)
        zeros = np.count_nonzero(self.X == 0)

        nobs_perrow = np.count_nonzero(self.X > -1, axis=1)
        nzero_perrow = np.count_nonzero(self.X == 0, axis=1)

        nobs_persite = np.count_nonzero(self.X > -1, axis=0)
        nzero_persite = np.count_nonzero(self.X == 0, axis=0)

        # print(self.X.shape)
        # print(f"Total cells: {total_cells}")
        # print(f"Unobserved {unobserved}")
        # print(f"No of zeros: {zeros}")
        print(f"Unobserved percent: {unobserved / total_cells * 100:.2f}%")
        print(f"Zero percent: {zeros / (total_cells - unobserved) * 100:.2f}%")

        print(f"Max number of observations per day: {np.max(nobs_perrow)}")
        print(f"Min number of observations per day: {np.min(nobs_perrow)}")

        print(f"Max number of observations per site: {np.max(nobs_persite)}")
        print(f"Min number of observations per site: {np.min(nobs_persite)}")

        observations = self.X[self.X > -1]
        unique, counts = np.unique(observations, return_counts=True)

        """
        figure(figsize=(8, 6), dpi=80)
        figure(figsize=(1,1)) would create an inch-by-inch image, which would be 80-by-80 pixels unless 
        you also give a different dpi argument.
        """
        figure(figsize=(8, 6), dpi=80)
        plt.bar(unique, counts)
        plt.title('Number of cots observed at the GBR')
        plt.xlabel('ncots')
        plt.ylabel('count')
        plt.ylim([0, 100])
        plt.xlim([0, 1000])

        plt.show()


if __name__ == "__main__":
    data_obj = GBRDataset(fileloc=dc.rawdata_loc,
                          poissonmodel=True,
                          fdim=2,
                          gat=False,
                          normalize=False,
                          savefeatures=True
                          )
    print("===============")
    print("Dataset statics")
    print("===============")
    data_obj.print_statistics()

    sid = 100
    tid = 300
    for item1, item2 in zip(data_obj.X[tid+1, :, 0], data_obj.Y[tid, :]):
        if item1 > 0 or item2 >0:
            print(item1, item2)
    print((data_obj.X[tid+1, :, 0]==data_obj.Y[tid, :]).sum())
    print(np.array_equal(data_obj.X[tid + 1, :, 0],
                         data_obj.Y[tid, :]
                         )
          )

    print("\n\n")
    for item1, item2 in zip(data_obj[tid+1]['x'][0], data_obj[tid]['y']):
        if item1 > 0 or item2 > 0:
            print(item1, item2)
    print(data_obj[tid+1]['x'], data_obj[tid]['y'])

    for ran in [100, 200, 300, 400, 500]:
        print(f'Front end sparsity: range({ran})')
        frontend = data_obj.X[:ran, :, 0]
        print(f"frontend shape: {frontend.shape}")
        print(f"number of observed cell: {len(frontend[frontend>-1])}, "
              f"ratio: {len(frontend[frontend>-1])/(frontend.shape[0]*frontend.shape[1]):0.4f}")

        print('Backend sparsity:')
        backend = data_obj.X[-ran:, :, 0]
        print(f"backend shape: {backend.shape}")
        print(f"number of observed cell: {len(backend[backend > -1])}, "
              f"ratio: {len(backend[backend > -1]) / (backend.shape[0] * backend.shape[1]):0.4f}")
        print('**'*30)