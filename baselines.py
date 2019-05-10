from sklearn.metrics import mean_absolute_error
import numpy as np
#code from https://benjlindsay.com/projects/movielens-project-part-2/
class PerformanceOverTimeValidator():
    """Validator that evaluates model performance over time. For each time
    period tested, all data before that time period is used as training data.
    
    Parameters
    ----------
    model : object
        An object with public fit(X, y) and predict(X) methods.
    err_func : function, default=sklearn.metrics.mean_absolute_error
        A function used to evaluate performance for each time period.
        Arguments are y and y_pred, both of which are array-like with
        shape = [n_samples]
    n_year_period : int, default=1
        Number of years per test period
        
    Attributes
    ----------
    model_ : object
        Model passed in parameters
    err_func_ : function
        Error function passed in parameters
    n_year_period_ : int
        n_year_period passed in parameters
    test_years_ : int list
        List of years, each of which marks the first year
        in a tested time period
    test_errs_ : float list, [n_periods_tested]
        List of test errors for each time period tested
    
    """
    def __init__(self, model, err_func=mean_absolute_error, n_year_period=1):
        self.model_ = model
        self.err_func_ = err_func
        self.n_year_period_ = n_year_period
        self.test_years_ = []
        self.test_errs_ = []
        
    def validate(self, X, y, years):
        """Computes test error using all previous data as training data
        over a set of years
        
        Paramters
        ---------
        X : {array-like}, shape = [n_samples, n_features]
            Feature vectors
        y : array-like, shape = [n_samples]
            Target variables
        years : array-like, shape = [n_samples]
        
        Returns
        -------
        test_years_ : int list, [n_periods_tested]
            List of years, each of which marks the first year
            in a tested time period
        test_errs_ : float list, [n_periods_tested]
            List of test errors for each time period tested
        """
        years = np.array(years)
        unique_years = np.unique(years)
        test_years = []
        test_errs = []
        # for year in unique_years[self.n_year_period_::self.n_year_period_]:
        #     train_inds = years < year
        #     test_inds = (years >= year) & (years < year + self.n_year_period_)
        splitter = TimeSeriesSplitByYear(n_years=2)
        for train_inds, test_inds in splitter.split(X, years_data=years):
            X_train, y_train = X[train_inds], y[train_inds]
            X_test, y_test = X[test_inds], y[test_inds]
            self.model_.fit(X_train, y_train)
            y_pred = self.model_.predict(X_test)
            err = self.err_func_(y_pred, y_test)
            year = years[test_inds][0]
            test_years.append(year)
            test_errs.append(err)
        self.test_years_ = test_years
        self.test_errs_ = test_errs
        return test_years, test_errs

class DampedUserMovieBaselineModel():
    """Baseline model that of the form mu + b_u + b_i,
    where mu is the overall average, b_u is a damped user
    average rating residual, and b_i is a damped item (movie)
    average rating residual. See eqn 2.1 of
    http://files.grouplens.org/papers/FnT%20CF%20Recsys%20Survey.pdf
    
    Parameters
    ----------
    damping_factor : float, default=0
        Factor to bring residuals closer to 0. Must be positive.
    
    Attributes
    ----------
    mu_ : float
        Average rating over all training samples
    b_u_ : pandas Series, shape = [n_users]
        User residuals
    b_i_ : pandas Series, shape = [n_movies]
        Movie residuals
    damping_factor_ : float, default=0
        Factor to bring residuals closer to 0. Must be positive.
    """
    def __init__(self, damping_factor=0):
        self.damping_factor_ = damping_factor
    
    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ----------
        X : DataFrame, shape = [n_samples, 2]
            DataFrame with columns 'userId', and 'movieId'
        y : array-like, shape = [n_samples]
            Target values (movie ratings)
        
        Returns
        -------
        self : object
        """
        X = X.copy()
        X['rating'] = y
        self.mu_ = np.mean(y)
        user_counts = X['userId'].value_counts()
        movie_counts = X['movieId'].value_counts()
        b_u = (
            X[['userId', 'rating']]
            .groupby('userId')['rating']
            .sum()
            .subtract(user_counts * self.mu_)
            .divide(user_counts + self.damping_factor_)
            .rename('b_u')
        )
        X = X.join(b_u, on='userId')
        X['movie_residual'] = X['rating'] - X['b_u'] - self.mu_
        b_i = (
            X[['movieId', 'movie_residual']]
            .groupby('movieId')['movie_residual']
            .sum()
            .divide(movie_counts + self.damping_factor_)
            .rename('b_i')
        )
        self.b_u_ = b_u
        self.b_i_ = b_i
    
    def predict(self, X):
        """Return rating predictions
        
        Parameters
        ----------


        X : DataFrame, shape = [n_samples, 2]
            DataFrame with columns 'userId', and 'movieId'
        
        Returns
        -------
        y_pred : array-like, shape = [n_samples]
            Array of n_samples rating predictions
        """
        X = X.copy()
        X = X.join(self.b_u_, on='userId').fillna(0)
        X = X.join(self.b_i_, on='movieId').fillna(0)
        return self.mu_ + X['b_u'] + X['b_i']