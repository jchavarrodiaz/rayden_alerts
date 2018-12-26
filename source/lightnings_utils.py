# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
import imageio
from scipy.spatial import distance_matrix
from constants import dt_cut_points

plt.style.use('bmh')
db_path = '../db/db_rayden.db'
table = 'data_bogota'
dt_sel = '10T'
dist_min = .075  # Minimum distance between centroids.
min_lightnings_sel = list(dt_cut_points.keys())[list(dt_cut_points.values()).index('Moderada')]


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def sel_storms_daily():
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    query = """
    select strftime('%Y-%m-%d', datetime) as fecha,
        count(datetime) as rayos,
        avg(amperage) as magnitud,
        min(datetime) as inicio,
        max(datetime) as fin
    from data_bogota
    where(longitude BETWEEN -74.3 AND -73.9 AND latitude BETWEEN 4.2 AND 4.9)
    group by strftime('%Y-%m-%d', datetime)
    order by rayos desc
    """

    cur.execute(query)
    results = cur.fetchall()
    columns = [i[0] for i in cur.description]

    df_storms = pd.DataFrame(data=results, columns=columns)
    df_storms['fecha'] = pd.to_datetime(df_storms['fecha'])
    df_storms['inicio'] = pd.to_datetime(df_storms['inicio'])
    df_storms['fin'] = pd.to_datetime(df_storms['fin'])

    return df_storms


def sel_storms_interval(seconds=600):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Query for extracting data within {0} seconds interval.
    query = """
    select datetime((strftime('%s', datetime) / {0}) * {0}, 'unixepoch') as fecha,
        count(*) as rayos,
        avg(amperage) as magnitud
    from data_bogota
    where(longitude BETWEEN -74.3 AND -73.9 AND latitude BETWEEN 4.2 AND 4.9)
    group by fecha
    order by fecha
    """.format(seconds)

    cur.execute(query)
    results = cur.fetchall()
    columns = [i[0] for i in cur.description]

    df_storms = pd.DataFrame(data=results, columns=columns)
    df_storms['fecha'] = pd.to_datetime(df_storms['fecha'])

    # TODO: Seleccionar intervalos consecutivos con espacios menores a una hora.

    return df_storms


def classify_storm(x, cut_points=None):
    """
    Classifies storm based on its number of lightnings.

    :param x: number of lightnings
    :type x: int
    :param cut_points:
    :type cut_points: dict
    :return: Categorical value for the storm
    :rtype: str
    """
    if not cut_points:
        cut_points = {5: 'Leve', 10: 'Baja', 12: 'Moderada', 24: 'Fuerte', 35: 'Muy Fuerte', 47: 'Extrema'}

    label = None

    for limit in sorted(cut_points, reverse=True):
        if x >= limit:
            label = cut_points[limit]
            break

    return label


def plot_lightnings(date_storm):
    print(date_storm)
    # date_storm = pd.to_datetime('2018-03-18')
    path_csv = '../results/{0:%Y%m%d}/{0:%Y%m%d}.csv'.format(date_storm)
    df_data = pd.read_csv(path_csv, index_col=0)
    df_data['datetime'] = pd.to_datetime(df_data['datetime'], format='%Y-%m-%d %H:%M:%S')
    df_data['interval'] = pd.to_datetime(df_data['interval'], format='%Y-%m-%d %H:%M')
    df_summary = df_data.groupby('interval').mean()
    df_summary['count'] = df_data.groupby(['interval']).count()['datetime']
    fig = plt.figure(figsize=[8, 14])  # longitude BETWEEN -74.3 AND -73.9 AND latitude BETWEEN 4.2 AND 4.9
    ax = fig.add_subplot(111)
    smap = ax.scatter(x=df_summary['longitude'], y=df_summary['latitude'], c=df_summary.index, cmap='Blues')
    plt.ylim([4.2, 4.9])
    plt.xlim([-74.3, -73.9])
    cbar = fig.colorbar(smap)
    cbar.ax.set_yticklabels(df_summary.index.strftime('%H:%M'))
    plt.title(u"Descargas eléctricas en Bogotá {:%Y-%m-%d}".format(date_storm))
    plt.tight_layout()
    fig_time = '../figs/{0:%Y%m%d}/{0:%Y%m%d}_time'.format(date_storm)
    plt.savefig(fig_time)
    plt.close()

    fig = plt.figure(figsize=[12, 7.5])  # longitude BETWEEN -74.3 AND -73.9 AND latitude BETWEEN 4.2 AND 4.9
    ax = fig.add_subplot(111)
    df_summary[['count', 'amperage']].plot(ax=ax, style='o-', secondary_y=['amperage'])
    plt.title(u"Descargas eléctricas en Bogotá {:%Y-%m-%d}".format(date_storm))
    # plt.set_ylabel('')
    plt.tight_layout()
    fig_time = '../figs/{0:%Y%m%d}/{0:%Y%m%d}_count'.format(date_storm)
    plt.savefig(fig_time)
    plt.close()


def plot_storm(date_storm, start, stop, dt='5T'):
    """

    :param date_storm:
    :param start:
    :param stop:
    :param dt:
    :return:
    """
    print(date_storm)
    folder_figs = '../figs/{:%Y%m%d}'.format(date_storm)
    create_folder(folder_figs)
    folder_results = '../results/{:%Y%m%d}'.format(date_storm)
    create_folder(folder_results)

    con = sqlite3.connect(db_path)

    idx_dt = pd.date_range(start, stop, freq=dt, name='Time')

    query = """
    SELECT *
    FROM {}
    WHERE datetime BETWEEN '{}' AND '{}'
    AND (longitude BETWEEN -74.3 AND -73.9 AND latitude BETWEEN 4.2 AND 4.9)
    ORDER BY datetime
    """.format(table, start, stop)

    cur = con.cursor()
    cur.execute(query)

    results = cur.fetchall()
    columns = [i[0] for i in cur.description]
    df_data = pd.DataFrame(data=results, columns=columns)
    df_data['datetime'] = pd.to_datetime(df_data['datetime'])
    df_data['interval'] = pd.to_datetime('NaT')

    dt_pre = idx_dt[0]
    # dt_pre = dt_pre.replace(day=10, hour=22, minute=15)
    title = u"Descargas eléctricas en Bogotá {:%Y-%m-%d}".format(date_storm)
    meridians = np.arange(-74.2, -73.9, .1)
    parallels = np.arange(4.3, 4.81, .1)

    images = []
    dt_radius = {}

    for interval in idx_dt[1:]:
        # interval = interval.replace(day=10, hour=22, minute=30)
        print(interval)
        df_interval = df_data[(dt_pre <= df_data['datetime']) & (df_data['datetime'] < interval)].copy()
        df_data.loc[df_interval.index, 'interval'] = interval
        dt_pre = interval

        df_cluster = df_interval[['longitude', 'latitude']].copy()
        n_lightnings = df_cluster.shape[0]

        if n_lightnings > 2:

            # model = KMeans(n_clusters=nclusters)
            model = DBSCAN(eps=dist_min, min_samples=10)
            model.fit(df_cluster)

            # Fitting Model
            model.fit(df_cluster)
            cluster_labels = model.fit_predict(df_cluster)
            df_cluster['cluster'] = cluster_labels
            dfg_cluster = df_cluster.groupby('cluster')
            df_centroids = dfg_cluster.mean()
            df_counts = dfg_cluster.count()
            df_cluster.reset_index(inplace=True, drop=True)

            dt_radius[interval] = [[interval, distance_matrix(x=pd.DataFrame(df_centroids.loc[i]).T,
                                                              y=df_cluster.loc[df_cluster['cluster'] == i,
                                                                               ['longitude', 'latitude']]).mean(),
                                    df_counts.loc[i, 'latitude']]
                                   for i in df_centroids.index]

            # Plot clustering
            plt.figure(figsize=[9, 10.5])
            m = Basemap(projection='cyl', llcrnrlat=4.2, urcrnrlat=4.9, llcrnrlon=-74.3, urcrnrlon=-73.9,
                        resolution='i')
            m.drawmeridians(meridians, linewidth=.2, labels=[False, False, False, True])
            m.drawparallels(parallels, linewidth=.2, labels=[True, True, False, False])
            x, y = m(df_cluster['longitude'], df_cluster['latitude'])
            m.scatter(x, y, c=df_cluster['cluster'])

        else:
            # Plot clustering
            plt.figure(figsize=[9, 10.5])
            m = Basemap(projection='cyl', llcrnrlat=4.2, urcrnrlat=4.9, llcrnrlon=-74.3, urcrnrlon=-73.9,
                        resolution='i')
            m.drawmeridians(meridians, linewidth=.2, labels=[False, False, False, True])
            m.drawparallels(parallels, linewidth=.2, labels=[True, True, False, False])

        m.readshapefile('../gis/ZonasIDIGER', 'ZonasIDIGER')
        plt.text(x=-74.07, y=4.88, s='{:%Y-%m-%d %H:%M} HLC'.format(interval), fontsize='x-large')
        plt.title(title)
        plt.tight_layout()
        namefig = '{}/{:%Y%m%d%H%M}'.format(folder_figs, interval)
        plt.savefig(namefig)
        plt.close()

        images.append(imageio.imread('{}.png'.format(namefig)))

    # Save gif
    imageio.mimsave('{}/{:%Y%m%d}.gif'.format(folder_figs, date_storm), images, duration=0.15, subrectangles=True)
    df_data.to_csv('{}/{:%Y%m%d}.csv'.format(folder_results, date_storm))

    # Number of lightnings and its radius
    radius = []

    for i in dt_radius:
        radius.extend(dt_radius[i])

    df = pd.DataFrame(radius)
    df.columns = ['Fecha', 'Radio', 'Cantidad']
    df.set_index('Fecha', inplace=True)

    return df


def main():
    df_storms = sel_storms_daily()
    df_radius = pd.DataFrame()

    for storm in df_storms.index[:50]:
        start = df_storms.loc[storm, 'inicio'] - pd.Timedelta(hours=2)
        start = start.replace(minute=0, second=0, microsecond=0)
        stop = df_storms.loc[storm, 'fin'] + pd.Timedelta(hours=2)
        stop = stop.replace(minute=55, second=0, microsecond=0)

        date_storm = (start + (stop - start) / 2).date()
        # plot_lightnings(date_storm)
        df = plot_storm(date_storm, start, stop, dt=dt_sel)
        df['Tormenta'] = storm
        df_radius = df_radius.append(df)

        df.sort_index(inplace=True)
        df.reset_index(inplace=True)
        df.plot.scatter(x='Cantidad', y='Radio', c='Fecha')
        plt.title('Cantidad y Radios Tormeta {:%Y-%m-%d}'.format(date_storm))
        plt.tight_layout()
        plt.savefig('../figs/{0:%Y%m%d}/{0:%Y%m%d}_radius'.format(date_storm))
        plt.close()

    df_radius.sort_index().sort_values('Tormenta', inplace=True)
    df_radius.to_excel('../results/radius.xlsx', 'Radios')
    df_radius.plot.scatter(x='Cantidad', y='Radio', c='Tormenta')
    plt.title('Cantidad y Radios de las tormentas seleccionadas')
    plt.tight_layout()
    plt.savefig('../figs/radius_total')
    plt.close()


if __name__ == '__main__':
    main()
    # df_storms = sel_storms_interval()
    # df_storms.to_excel('../results/storms_daily.xlsx', 'storms')
    # plot_lightnings()
    pass
