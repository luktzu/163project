"""
Lukas Metzner
CSE 163 AH
This script allows the user to input a desired team, window size, and max tree
depth from which to make a prediction of the team's next game outcome
"""
from predict_outcome import create_gamelog, create_windows_df, \
    predict_game_outcome


def city_gamelog():
    """
    creates a gamelog df from user's input. if input is not valid, loops
    until a valid input is given
    returns: gamelog dataframe with input city's team's games from this season
    """
    gamelog_df = None
    city = None
    print("This program will predict the outcome of a team's next game")
    print("Please enter the city of the team to predict:")
    city = input("> ")
    try:
        gamelog_df = create_gamelog(city, '2020-21')
    except IndexError:
        print("\nSomething is wrong with your city name. Please try again")
        print("restarting...\n---------------------------------------------\n")
        gamelog_df = city_gamelog()
    return gamelog_df


def get_winsize(dflen):
    """
    gets a desired window size from the user. if input is not a valid
    number, or if it is too high (more than half of games played),
    loops until a valid input is given.
    returns: int with user window size input, 10 by default if user
    enters nothing
    """
    print(f"\nPlease enter the sample size of recent games on which"
          " to base the prediction.\n(must not me more than"
          " half the number of games played so far)\nGames played:",
          dflen, '\nDefault sample size: 10')
    win_size = input('> ')
    if win_size == '':
        return 10
    try:
        win_size = int(win_size)
    except ValueError:
        print('\nPlease input a valid number')
        win_size = get_winsize(dflen)
    if win_size > dflen / 2:
        print('\nToo many games, please try again')
        win_size = get_winsize(dflen)
    return win_size


def get_depth():
    """
    gets a maximum depth from user. if input is not a valid number,
    loops until a valid input is given.
    returns: int with user depth input. None by default if user inputs
    nothing
    """
    print("\nPlease enter the maximum depth for the Decision Tree"
          " Classifier for the prediction. Default depth: None")
    depth = input("> ")
    if depth == '':
        return None
    try:
        depth = int(depth)
    except ValueError:
        print('\nPlease input a valid number')
        depth = get_depth()
    return depth


def main():
    gamelog_df = city_gamelog()
    win_size = get_winsize(len(gamelog_df))
    depth = get_depth()
    win_df = create_windows_df(gamelog_df, win_size)
    print('\n\nPredicting game outcome...\n\n')
    prediction = predict_game_outcome(win_df.loc[[0]], win_df, depth)
    print('Predicted outcome:', prediction)


if __name__ == "__main__":
    main()
