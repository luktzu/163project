"""
Lukas Metzner
CSE 163 AH
This script takes statistical data from nba.com to create Decision
Tree Classifers to predict game outcomes from running averages of
previous recent game statistical performance. Using historical data
from five diverse nba teams' seasons, it generates a number of trees
to assess the best recent game sample size and maximum tree depth for
such a model, and plots these. It also plots the most and least
important statistical parameters in making these predictive trees
"""
from nba_api.stats.endpoints import TeamGameLog
from nba_api.stats.static import teams
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def create_gamelog(city, season):
    """
    takes a city and season, and returns a pandas dataframe containing
    team statistical performance in major categories from that season
    returns an error if the city or season are unlisted or out of range
    parameters:
    city: str of name of city of the team of interest
    season: str of season of interest; format: "yyYY-{YY+1}", eg '2010-11'
    returns:
    pandas dataframe game log of the season with relevant stats and info
    """
    my_team = teams.find_teams_by_city(city)[0]
    my_team_id = my_team['id']
    team_season = TeamGameLog(team_id=my_team_id, season=season)
    gamelog_df = team_season.get_data_frames()[0]
    # note: the games are listed in reverse order, such that the most recent
    # game, the last game of the season, is at row 0.
    gamelog_df['GAME_DATE'] = pd.to_datetime(gamelog_df['GAME_DATE'],
                                             format='%b %d, %Y')
    return gamelog_df


def create_windows_df(gamelog, win_size):
    """
    takes a given game log and window size, and creates a new dataframe
    containing running averages in relevant statistical categories for each
    game after the win_size-th game of the season. win percentage only
    accounts for the games in the window.
    parameters:
    gamelog: pandas dataframe of game logs from one team's games in a season
    win_size: int of desired previous recent game sample size, from which to
    generate running average statistics
    returns:
    pandas dataframe with relevant info and stats for each game, with the
    aforementioned categories manipulated to represent recent performance
    """
    log_columns = []
    for column in gamelog.columns:
        log_columns.append(column)
    feature_stats = log_columns[8:]
    df_win = gamelog.copy()
    # creates 10-game rolling mean for each feature stat in the win dataframe
    for stat in feature_stats:
        df_win[stat] = gamelog.loc[:, stat].rolling(window=win_size
                                                    ).mean().shift(-win_size)

    # creates 10-game running W-L counts and win% for each game past win_size
    for i in range(len(gamelog) - win_size):
        window_wins = np.count_nonzero(gamelog.loc[i+1:i+win_size,
                                                   'WL'] == 'W')
        # df_win.loc[i, 'W'] = window_wins
        # df_win.loc[i, 'L'] = np.count_nonzero(gamelog.loc[i+1:i+win_size,
        #                                                   'WL'] == 'L')
        df_win.loc[i, 'W_PCT'] = window_wins / win_size
    df_win = df_win.drop(['W', 'L'], axis=1)
    # df_win.loc[len(df):len(df) - win_size, ['W', 'L']] = float('nan')
    # not needed since these rows get dropped later anyway

    # create a column for "days since last game"
    df_win['days_since_game'] = -(df_win['GAME_DATE'].diff().shift(-1).apply
                                  (lambda x: x.days))

    # tests rolling mean manually for one game
    # print(df_win.loc[win_size, ['GAME_DATE', 'PTS']])
    # print(df.loc[win_size, 'GAME_DATE'])
    # print(df.loc[win_size+1:win_size*2, 'PTS'].mean())
    return df_win


def test_dtc(df_win, depth=None):
    """
    trains and tests a Decision Tree Classifier to predict outcomes
    of games using a table of recent running statistical averages, using
    performance stats as features, and game outcome (W/L) as labels
    splits data into 70/30 training/testing data
    parameters:
    df_win: pandas dataframe with relevant info and recent game running
    averages in statistical categories for each game of a season
    depth: maximum DecisionTreeClassifer depth. default sets no maximum
    returns:
    train_acc: float between 0 and 1, accuracy score of model on predictions
    of training data labels and actual training data labels
    train_acc: float between 0 and 1, accuracy score of model on predictions
    of test data labels and actual test data labels
    model: scikit DecisionTreeClassifier object of actual predictive model
    """
    df_win = df_win.dropna()
    df_win = df_win.drop(['Team_ID', 'Game_ID', 'GAME_DATE',
                          'MATCHUP'], axis=1)
    labels = df_win['WL']
    features = df_win.loc[:, df_win.columns != 'WL']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3)
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(features_train, labels_train)
    train_pred_labels = model.predict(features_train)
    test_pred_labels = model.predict(features_test)
    train_acc = accuracy_score(labels_train, train_pred_labels)
    test_acc = accuracy_score(labels_test, test_pred_labels)
    return train_acc, test_acc, model


def predict_game_outcome(game_running_avg, base_season_df, depth=None):
    """
    predicts outcome of a single game from running average stats prior to
    that game. bases ML model on a particular base season's pre-calculated
    running average stats dataframe.
    parameters:
    game_running_avg: series, with average stats from games prior to game
    base_season_df: pre-computed running average df from game logs
    depth: int, representing max depth of decision tree classifier to be
    used, default value 2.
    returns: str, 'W' for predicted win or 'L' for predicted loss
    """
    model = test_dtc(base_season_df, depth)[2]
    game_running_features = game_running_avg.drop(['Team_ID', 'Game_ID',
                                                   'GAME_DATE', 'MATCHUP',
                                                   'WL'], axis=1)
    return model.predict(game_running_features)[0]


def assess_win_size(seasons):
    """
    creates a plot comparing training and testing data accuracy scores
    for models with no depth constraint, created with data from dataframes
    across window sizes (for sample size of recent game performance). assesses
    accuracies of 10 models for each team-season, for a total of 50 models, at
    each window size from 2 to 40 (inclusive) at steps of 2
    parameters:
    seasons: pandas dataframe with game logs with relevant info and stats of a
    team over a season
    returns:
    None, but saves scatterplot with window size on the x axis, and accuracy
    score on the y axis. plots the training and test accuracies in distinct
    colors, indicated on legend
    """
    output_train_accs = []
    output_test_accs = []
    # outer loop goes 14 times
    for win_size in range(2, 41, 4):
        size_train_accs = []
        size_test_accs = []
        # seasons loops 5 times, range is 10, 50 total for each size
        for season in seasons:
            df_win = create_windows_df(season, win_size)
            for i in range(10):
                train_acc, test_acc = test_dtc(df_win)[0:2]
                size_train_accs.append(train_acc)
                size_test_accs.append(test_acc)
        # print('win size:', win_size)
        avg_train_acc = sum(size_train_accs) / 50
        avg_test_acc = sum(size_test_accs) / 50
        # print('train acc:', avg_train_acc, '\ntest acc', avg_test_acc)
        output_train_accs.append(avg_train_acc)
        output_test_accs.append(avg_test_acc)
    fig, ax = plt.subplots(1)
    ax.scatter(range(2, 41, 4), output_train_accs, label='Train Accuracy')
    ax.scatter(range(2, 41, 4), output_test_accs, label='Test Accuracy')
    ax.set_xlabel('Window Size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Prediction Accuracy vs. Window Size')
    ax.legend()
    plt.savefig('win_size.png')
    plt.close()


def assess_tree_depth(seasons):
    """
    creates a plot comparing training and testing data accuracy scores
    for models created with 10-game running average dataframes across maximum
    tree depths. assesses accuracies of 30 models for each team-season, for a
    total of 150 models, at each maximum tree depth between 1 and 10
    parameters:
    seasons: pandas dataframe with game logs with relevant info and stats of a
    team over a season
    returns:
    None, but saves scatterplot with max depth on the x axis, and accuracy
    score on the y axis. plots the training and test accuracies in distinct
    colors, indicated on legend
    """
    output_train_acc = []
    output_test_acc = []
    for depth in range(1, 11):
        depth_train_acc = []
        depth_test_acc = []
        for season in seasons:
            df_win = create_windows_df(season, 10)
            for i in range(30):
                train_acc, test_acc = test_dtc(df_win, depth)[0:2]
                depth_train_acc.append(train_acc)
                depth_test_acc.append(test_acc)
        # print('depth:', depth)
        avg_train_acc = sum(depth_train_acc) / 150
        avg_test_acc = sum(depth_test_acc) / 150
        # print('train acc:', avg_train_acc, '\ntest acc', avg_test_acc)
        output_train_acc.append(avg_train_acc)
        output_test_acc.append(avg_test_acc)
    fig, ax = plt.subplots(1)
    ax.scatter(range(1, 11), output_train_acc, label='Train Accuracy')
    ax.scatter(range(1, 11), output_test_acc, label='Test Accuracy')
    ax.set_xlabel('Tree Depth')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Prediction Accuracy vs. Decision Tree Depth')
    ax.legend()
    plt.savefig('tree_depth.png')
    plt.close()


def assess_features(seasons, n=6, depth=None):
    """
    plots the top n and bottom n statistical features in terms of importance
    to a large number of DecisionTreeClassifiers across team-seasons, on data
    from last 10 games. assesses importance of each feature across 300 models
    generated for each team-season, for a total of 1500 models.
    parameters:
    seasons: pandas dataframe with game logs with relevant info and stats of a
    team over a season
    n: number of top/bottom statistical categories to plot
    depth: max depth parameter for trees in assessment
    returns:
    None, but saves a bar chart with two set of axes: on top, the top 6 most
    important statistical parameters across the models. on bottom, bottom 6
    least important statistical parameters. y-axis shows 'importance',
    a relative value generated through the scikit library's inbuilt methods
    """
    importance_list = []
    features = create_windows_df(seasons[0], 10)
    features = features.drop(['Team_ID', 'Game_ID', 'GAME_DATE',
                              'MATCHUP', 'WL'], axis=1)
    for feature in features.columns:
        importance_list.append([feature, 0])
    for season in seasons:
        df_win = create_windows_df(season, 10)
        for i in range(300):
            model = test_dtc(df_win, depth)[2]
            for idx, importance in zip(range(len(features.columns)),
                                       model.feature_importances_):
                importance_list[idx][1] += importance
    for feature in importance_list:
        feature[1] /= 1500
    importance_list.sort(key=lambda x: x[1], reverse=True)
    most_important = importance_list[0:n]
    least_important = importance_list[(-n-1):-1]
    fig, axs = plt.subplots(2, sharey=True)
    axs[0].bar([x[0] for x in most_important], [x[1] for x in most_important])
    axs[0].set_xlabel('Statistical Feature')
    axs[0].set_ylabel('Importance')
    axs[0].set_title('Most Important Statistical Features')
    axs[1].bar([x[0] for x in least_important],
               [x[1] for x in least_important])
    axs[1].set_xlabel('Statistical Feature')
    axs[1].set_ylabel('Importance')
    axs[1].set_title('Least Important Statistical Features')
    plt.tight_layout()
    plt.savefig('importance_.png')
    plt.close()


def plot_sample_tree(seasons):
    """
    plots an example Decision Tree Classifier, for visualization
    of how these models work. uses 10-game window data from the 2005-06
    Houston Rockets Season, and sets no maximum tree depth.
    parameters:
    seasons: pandas dataframe with game logs with relevant info and stats of a
    team over a season
    returns:
    None, but saves a plotted example tree. this tree is nearly impossible to
    interpret because the features are not labeled as stats, but can be useful
    to reference general structure and verify most common maximum depth for
    a fully-fitted tree is 7
    """
    df_win = create_windows_df(seasons[3], 10)
    plot_tree(test_dtc(df_win)[2])
    plt.savefig('sample_tree.png')
    plt.close()


def main():
    five_seasons = [('chicago', '2010-11'), ('philadelphia', '2015-16'),
                    ('golden state', '2015-16'), ('houston', '2005-06'),
                    ('memphis', '2004-05')]
    seasons_gamelogs = [create_gamelog(*season) for season in five_seasons]
    # generates one tree and displays test and train accuracy for 2010-11 bulls
    bulls_train_acc, bulls_test_acc = test_dtc(seasons_gamelogs[0])[0:2]
    print('test accuracy on chicago 2010-11:', bulls_test_acc, '\ntrain'
          ' accuracy on chicago 2010-11', bulls_train_acc)

    assess_win_size(seasons_gamelogs)
    assess_tree_depth(seasons_gamelogs)
    assess_features(seasons_gamelogs)
    plot_sample_tree(seasons_gamelogs)


if __name__ == "__main__":
    main()
