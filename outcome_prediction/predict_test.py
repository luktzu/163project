"""
Lukas Metzner
CSE 163 AH
This script runs some basic tests on the game log dataframes generated in the
main script and compares their values with manually verified or computed values
to ensure that the outputs are correct
"""

from cse163_utils import assert_equals
from predict_outcome import create_gamelog, create_windows_df


def test_gamelog():
    """
    compares manually searched stats from some games in the miami heat 2019-20
    season and compares to data retrieved from the dataframe to assert equality
    """
    heat_gamelog = create_gamelog('miami', '2019-20')
    assert_equals(heat_gamelog.loc[0, ['PTS']][0], 92)
    assert_equals(heat_gamelog.iloc[-1, -1], 120)
    assert_equals(heat_gamelog.loc[0, ['WL']][0], 'L')


def test_windows_df():
    """
    runs a few assert statements on running-window stats computed with the
    create windows df function to computer-calculated and manually-calculated
    stats from the same stretches, with the same heat 2019-20 season
    """
    heat_gamelog = create_gamelog('miami', '2019-20')
    heat_windows_10 = create_windows_df(heat_gamelog, 10)
    assert_equals(heat_gamelog.loc[1:10, ['REB']].mean()[0],
                  heat_windows_10.loc[0, ['REB']][0])
    assert_equals(0.5, heat_windows_10.loc[1, ['W_PCT']][0])
    assert_equals(heat_gamelog.loc[13:22, ['AST']].mean()[0],
                  heat_windows_10.loc[12, ['AST']][0])
    manual_pts_loc12 = sum([116, 126, 126, 119, 124, 124, 101, 113, 109, 97])
    assert_equals(manual_pts_loc12 / 10, heat_windows_10.loc[12, ['PTS']][0])


def main():
    test_gamelog()
    test_windows_df()


if __name__ == "__main__":
    main()
