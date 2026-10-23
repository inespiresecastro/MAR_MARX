# ==========================================================
# PSY BUBBLE DETECTION
# Adapted from PSYMONITOR R Package
# Phillips, Shi and Yu (2015)
# ==========================================================

import numpy as np
import pandas as pd


def ADF(y, IC=0, adflag=0):
    """
    Exact Python port of psymonitor::ADF()

    Parameters
    ----------
    y : array_like
    IC : int
        0 = fixed lag
        1 = AIC
        2 = BIC
    adflag : int
        lag order (IC=0)
        maximum lag (IC>0)

    Returns
    -------
    float
        Right-tailed ADF t-statistic
    """

    y = np.asarray(y, dtype=float)

    T0 = len(y)
    T1 = T0 - 1

    const = np.ones(T1)

    y1 = y[:-1]
    y0 = y[1:]

    dy = y0 - y1

    x1 = np.column_stack((y1, const))

    t = T1 - adflag
    dof = t - adflag - 2

    if dof <= 0:
        return np.nan

    if IC > 0:

        ICC = np.zeros(adflag + 1)

        ADFstats = np.zeros(adflag + 1)

        for k in range(adflag + 1):

            xx = x1[k:, :]

            dy01 = dy[k:]

            if k > 0:

                x2 = xx.copy()

                for j in range(1, k + 1):

                    lagged = dy[k - j:T1 - j]

                    x2 = np.column_stack((x2, lagged))

            else:

                x2 = xx

            beta = np.linalg.solve(
                x2.T @ x2,
                x2.T @ dy01
            )

            eps = dy01 - x2 @ beta

            npdf = np.sum(
                -0.5 * np.log(2 * np.pi)
                - 0.5 * eps**2
            )

            if IC == 1:

                ICC[k] = (
                    -2 * npdf / t
                    + 2 * len(beta) / t
                )

            else:

                ICC[k] = (
                    -2 * npdf / t
                    + len(beta) * np.log(t) / t
                )

            sigma2 = (
                eps.T @ eps
            ) / dof

            cov = (
                sigma2
                * np.linalg.inv(
                    x2.T @ x2
                )
            )

            se = np.sqrt(
                np.diag(cov)
            )

            ADFstats[k] = (
                beta[0] / se[0]
            )

        return ADFstats[
            np.argmin(ICC)
        ]

    # fixed lag

    xx = x1[adflag:, :]

    dy01 = dy[adflag:]

    if adflag > 0:

        x2 = xx.copy()

        for j in range(
            1,
            adflag + 1
        ):

            lagged = dy[
                adflag - j:T1 - j
            ]

            x2 = np.column_stack(
                (
                    x2,
                    lagged
                )
            )

    else:

        x2 = xx

    beta = np.linalg.solve(
        x2.T @ x2,
        x2.T @ dy01
    )

    eps = dy01 - x2 @ beta

    sigma2 = (
        eps.T @ eps
    ) / dof

    cov = (
        sigma2
        * np.linalg.inv(
            x2.T @ x2
        )
    )

    se = np.sqrt(
        np.diag(cov)
    )

    return beta[0] / se[0]

def ADFres(
    y,
    IC=0,
    adflag=0
):
    """
    Exact port of psymonitor::ADFres()
    """

    y = np.asarray(
        y,
        dtype=float
    )

    T0 = len(y)

    T1 = T0 - 1

    dy = np.diff(y)

    const = np.ones(T1)

    x1 = const.reshape(-1, 1)

    t = T1 - adflag

    if IC > 0:

        ICC = np.zeros(
            adflag + 1
        )

        betas = []

        residuals = []

        for k in range(
            adflag + 1
        ):

            xx = x1[k:, :]

            dy01 = dy[k:]

            if k > 0:

                x2 = xx.copy()

                for j in range(
                    1,
                    k + 1
                ):

                    lagged = dy[
                        k - j:T1 - j
                    ]

                    x2 = np.column_stack(
                        (
                            x2,
                            lagged
                        )
                    )

            else:

                x2 = xx

            beta = np.linalg.solve(
                x2.T @ x2,
                x2.T @ dy01
            )

            eps = (
                dy01
                - x2 @ beta
            )

            betas.append(beta)

            residuals.append(eps)

            npdf = np.sum(
                -0.5 * np.log(
                    2 * np.pi
                )
                - 0.5 * eps**2
            )

            if IC == 1:

                ICC[k] = (
                    -2 * npdf / t
                    + 2 * len(beta) / t
                )

            else:

                ICC[k] = (
                    -2 * npdf / t
                    + len(beta)
                    * np.log(t)
                    / t
                )

        best = np.argmin(
            ICC
        )

        return {
            "beta": betas[best],
            "eps": residuals[best],
            "lag": best
        }

    xx = x1[
        adflag:
    ]

    dy01 = dy[
        adflag:
    ]

    if adflag > 0:

        x2 = xx.copy()

        for j in range(
            1,
            adflag + 1
        ):

            lagged = dy[
                adflag - j:T1 - j
            ]

            x2 = np.column_stack(
                (
                    x2,
                    lagged
                )
            )

    else:

        x2 = xx

    beta = np.linalg.solve(
        x2.T @ x2,
        x2.T @ dy01
    )

    eps = (
        dy01
        - x2 @ beta
    )

    return {
        "beta": beta,
        "eps": eps,
        "lag": adflag
    }




def PSY(
    y,
    swindow0=None,
    IC=0,
    adflag=0
):
    """
    Exact port of psymonitor::PSY()

    Returns
    -------
    bsadf : ndarray
        BSADF sequence
    """

    y = np.asarray(y, dtype=float)

    T = len(y)

    if swindow0 is None:
        swindow0 = int(
            np.floor(
                T * (0.01 + 1.8 / np.sqrt(T))
            )
        )

    bsadfs = np.full(T, np.nan)

    for r2 in range(
        swindow0,
        T + 1
    ):

        rwadf = np.full(
            r2 - swindow0 + 1,
            -999.0
        )

        for r1 in range(
            r2 - swindow0 + 1
        ):

            rwadf[r1] = ADF(
                y[r1:r2],
                IC=IC,
                adflag=adflag
            )

        bsadfs[r2 - 1] = np.max(rwadf)

    return bsadfs



def locate(index, dates):
    index = np.asarray(index, dtype=int)
    dates = pd.to_datetime(dates)

    if index.size == 0 or index.max() == 0:
        return []

    episodes = []
    in_episode = False

    for i, value in enumerate(index):
        if value == 1 and not in_episode:
            start = dates[i]
            in_episode = True

        elif value == 0 and in_episode:
            episodes.append((start, dates[i - 1]))
            in_episode = False

    if in_episode:
        episodes.append((start, dates[-1]))

    return episodes




def cvPSYwmboot(
    y,
    swindow0=None,
    IC=0,
    adflag=0,
    Tb=30,
    nboot=199
):
    y = np.asarray(y, dtype=float).flatten()
    T = len(y)

    if swindow0 is None:
        swindow0 = int(
            np.floor(
                T * (
                    0.01
                    + 1.8 / np.sqrt(T)
                )
            )
        )

    swindow0 = int(swindow0)

    if Tb < swindow0:
        raise ValueError(
            f"Tb ({Tb}) must be greater than or equal to "
            f"swindow0 ({swindow0})."
        )

    result = ADFres(
        y,
        IC=IC,
        adflag=adflag
    )

    beta = np.asarray(result["beta"])
    eps = np.asarray(result["eps"]).flatten()
    lag = int(result["lag"])

    dy = np.diff(y)
    T0 = len(eps)

    np.random.seed(101)

    rN = np.random.randint(
        0,
        T0,
        size=(Tb, nboot)
    )

    wn_value = np.random.normal()

    wn = np.full(
        (Tb, nboot),
        wn_value
    )

    dyb = np.zeros(
        (Tb - 1, nboot)
    )

    if lag > 0:
        dyb[:lag, :] = dy[:lag].reshape(-1, 1)

    for j in range(nboot):

        if lag == 0:

            for i in range(Tb - 1):
                dyb[i, j] = (
                    wn[i, j]
                    * eps[rN[i, j]]
                )

        else:

            for i in range(lag, Tb - 1):

                x = np.zeros(lag)

                for k in range(1, lag + 1):
                    x[k - 1] = dyb[i - k, j]

                ar_part = np.dot(
                    x,
                    beta[1:].flatten()
                )

                innovation = (
                    wn[i - lag, j]
                    * eps[rN[i - lag, j]]
                )

                dyb[i, j] = (
                    ar_part
                    + innovation
                )

    yb = np.zeros(
        (Tb, nboot)
    )

    for j in range(nboot):
        yb[0, j] = y[0]
        yb[1:, j] = (
            y[0]
            + np.cumsum(dyb[:, j])
        )

    dim_psy = Tb - swindow0 + 1

    MPSY = np.empty(
        (nboot, dim_psy)
    )

    for b in range(nboot):

        stat = PSY(
            yb[:, b],
            swindow0=swindow0,
            IC=IC,
            adflag=adflag
        )

        stat = np.asarray(stat).flatten()

        if len(stat) < dim_psy:
            raise ValueError(
                f"PSY returned only {len(stat)} "
                f"observations, but {dim_psy} "
                f"were expected."
            )

        stat = stat[-dim_psy:]

        MPSY[b, :] = stat

    SPSY = np.nanmax(
        MPSY,
        axis=1
    )

    cv = np.quantile(
        SPSY,
        [0.90, 0.95, 0.99]
    )

    return cv



