import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import calendar
from datetime import date
from datetime import datetime,  timedelta, time
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from math import sin, cos, sqrt, atan2, radians
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import haversine_distances
import time as tm
import gspread
from oauth2client.service_account import ServiceAccountCredentials


# Set Config dan icon
def header():
    st.set_page_config(
            page_title='Routing Recommendation',
            layout='wide',
            initial_sidebar_state='expanded'
            )
    # Hide Streamlit Style
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    # Streamlit Markdown
    st.markdown("<h1 style='color:black;text-align: center'>Routing Recommendation</h1>", unsafe_allow_html=True)
    return header

def set_bg_hack_url():
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://i.ibb.co/3CWwXtN/output-onlinepngtools-1.png");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def vehicle_restriction(order, vehicle, transporter_route):
    # Membuat list transporter yang tersedia
    transporter_notlist = vehicle[~(vehicle['nama_transporter'].isin(transporter_route['nama_transporter']))].reset_index(drop= True)
    transporter_listed = transporter_route[transporter_route['nama_transporter'].isin(vehicle['nama_transporter'])].reset_index(drop= True)
    list_vehicle =pd.DataFrame()
    order_routing = pd.DataFrame(columns=order.columns)
    order_notrouting = pd.DataFrame(columns=order.columns)

    # Get list combination vehicle yang dapat di-assign ke order
    for index, row in order.iterrows():
        vehicle_type = row['jenis_kendaraan']
        constraint_tahun = int(row['constraint_tahun'])
        kota_asal = row['kota_asal']
        kota_tujuan = row['kota_tujuan']
        # Jika gaada constraint tahun tp ada constraint dedicated
        if constraint_tahun == 0 :
            temp_vehicle = vehicle[vehicle['vehicle_type'] == vehicle_type]
            temp_transporter = transporter_listed[(transporter_listed['nama_transporter'].isin(temp_vehicle['nama_transporter'])) & (transporter_listed['asal_kota'] == kota_asal) & (transporter_listed['tujuan_kota'] == kota_tujuan) & (transporter_listed['jenis_kendaraan'] == vehicle_type)]
            temp_transporter_nlist = transporter_notlist[transporter_notlist['vehicle_type'] == vehicle_type]
            temp_vehicle2 = temp_vehicle[temp_vehicle['nama_transporter'].isin(temp_transporter['nama_transporter'])]
            temp_list_vehicle = pd.concat([temp_vehicle2, temp_transporter_nlist], axis=0).reset_index(drop=True)
            if len(temp_vehicle) > 0 and (len(temp_transporter) > 0 or len(temp_transporter_nlist) > 0):
                order_routing.loc[len(order_routing)] = order.loc[index]
                temp_list_vehicle['id_order'] = row['id_order']
                list_vehicle = pd.concat([list_vehicle, temp_list_vehicle], axis=0).reset_index(drop = True)
            else :
                order_notrouting.loc[len(order_notrouting)] = order.loc[index]
        # Ketika ada constrain tahun dan constraint dedicated
        elif constraint_tahun > 0 :
            temp_vehicle = vehicle[(vehicle['vehicle_type'] == vehicle_type) & (vehicle['tahun_kendaraan'] >= constraint_tahun)]
            temp_transporter = transporter_listed[(transporter_listed['nama_transporter'].isin(temp_vehicle['nama_transporter'])) & (transporter_listed['asal_kota'] == kota_asal) & (transporter_listed['tujuan_kota'] == kota_tujuan) & (transporter_listed['jenis_kendaraan'] == vehicle_type)]
            temp_transporter_nlist = transporter_notlist[(transporter_notlist['vehicle_type'] == vehicle_type) & (transporter_notlist['tahun_kendaraan'] >= constraint_tahun)]
            temp_vehicle2 = temp_vehicle[temp_vehicle['nama_transporter'].isin(temp_transporter['nama_transporter'])]
            temp_list_vehicle = pd.concat([temp_vehicle2, temp_transporter_nlist], axis=0).reset_index(drop=True)
            if len(temp_vehicle) > 0 and (len(temp_transporter) > 0 or len(temp_transporter_nlist) > 0):
                order_routing.loc[len(order_routing)] = order.loc[index]
                temp_list_vehicle['id_order'] = row['id_order']
                list_vehicle = pd.concat([list_vehicle, temp_list_vehicle], axis=0).reset_index(drop = True)
            else :
                order_notrouting.loc[len(order_notrouting)] = order.loc[index]
        else :
            order_notrouting.loc[len(order_notrouting)] = order.loc[index]
    if len(list_vehicle) > 0:
        # Create master combination (join dengan order dan vehicle)
        master = pd.merge(list_vehicle[['nopol', 'id_order']], order, how = 'left', on = 'id_order').merge(vehicle, how ='left', on = 'nopol')
        master['unique_id'] = master['nopol'] + master['id_order']
    else :
        master = pd.DataFrame()
    return master

def recommendation_vehicle(vehicle_combination):
    """ Memberikan rekomendasi kendaraan terdekat dari order berdasarkan latitude & longitude
    Args:
        order : pandas dataframe list order not planned
        vehicle : pandas dataframe list vehicle idle
        vehicle_combination : pandas dataframe kombinasi kendaraan yang dapat di-assign pada order
    Returns:
        rekomendasi_vehicle_top3 : rekomendasi 3 kendaraan terdekat

    """

    df_concat = pd.DataFrame()
    list_idorder = vehicle_combination['id_order'].unique().tolist()

    # Looping setiap order untuk mendapatkan distance vehicle-order
    for id_order in list_idorder:
        # Filter order & vehicle sesuai jenis kendaraan
        temp_combination = vehicle_combination[vehicle_combination['id_order'] == id_order].reset_index(drop = True)
        
        # Get combination distance vehicle - order
        c1 = np.radians(temp_combination[['latitude_last','longitude_last']].to_numpy())
        c2 = np.radians(temp_combination[['latitude_asal','longitude_asal']].to_numpy())
        dist = haversine_distances(c2, c1) * 6371000/1000 * 1.3 # 1.3 adalah rasio dari euclidean : openstreetmap
        df = pd.DataFrame(dist, columns=temp_combination['nopol'], index=temp_combination['id_order'])
        df = df.reset_index()
        df.columns.name = None
        df = df.melt(id_vars=["id_order"], 
        var_name="nopol", 
        value_name="distance")
        
        # Return combination with vehicle & order information
        df_order_merge = temp_combination[['id_order', 'asal', 'tujuan', 'jenis_kendaraan']]
        df_vehicle_merge = temp_combination[['nopol', 'nama_transporter']]
        df = pd.merge(df, df_order_merge, how='left', on= 'id_order')
        df = pd.merge(df, df_vehicle_merge, how='left', on= 'nopol')
        df = df[['id_order', 'asal', 'tujuan', 'jenis_kendaraan','nopol', 'nama_transporter', 'distance']].sort_values(by= 'distance', ascending= True).reset_index(drop=True)
        df['ranking'] = df.groupby('id_order')['distance'].rank('dense')

        # Concat all order information
        df_concat = pd.concat([df_concat, df], axis = 0).reset_index(drop = True)
    
    # Memastikan kendaraan tidak di rekomendasikan lebih dari 1x (based on jarak terdekat)
    rekomendasi_vehicle = pd.DataFrame()
    for i in range(1,4):
        temp_df = df_concat.copy()
        if i  > 1 :
            temp_df = temp_df[(~temp_df['nopol'].isin(rekomendasi_vehicle['nopol']))].reset_index(drop = True)
        temp_vehicle = pd.DataFrame()
        while len(temp_df) > 0 :
            # temp_df = temp_df.sort_values(by= ['nopol', 'distance']).reset_index(drop = True)
            temp_df['rank_distance'] = temp_df.groupby('nopol')['distance'].rank('dense')
            # temp_df = temp_df.sort_values(by= ['id_order', 'distance']).reset_index(drop = True)
            temp_df['ranking'] = temp_df.groupby('id_order')['distance'].rank('dense')
            temp_df = temp_df.sort_values(by= ['id_order', 'distance']).reset_index(drop = True)
            subset = temp_df[temp_df['rank_distance'] == 1].drop_duplicates(subset = ['nopol'], keep='first').drop_duplicates(subset = ['id_order'], keep='first')
            temp_df = temp_df[(~temp_df['id_order'].isin(subset['id_order'])) & (~temp_df['nopol'].isin(subset['nopol']))].reset_index(drop = True)
            temp_vehicle = pd.concat([temp_vehicle, subset]).reset_index(drop = True)
        temp_vehicle['ranking'] = i
        
        # rekomendasi_vehicle = pd.concat([rekomendasi_vehicle, temp_vehicle]).reset_index(drop = True)
        rekomendasi_vehicle = pd.concat([rekomendasi_vehicle, temp_vehicle], axis= 0)
        rekomendasi_vehicle['ranking'] = rekomendasi_vehicle.groupby('id_order')['distance'].rank('dense')
        rekomendasi_vehicle = rekomendasi_vehicle.sort_values(by=['id_order', 'ranking']).reset_index(drop = True)
    rekomendasi_vehicle_top3 = rekomendasi_vehicle.drop(columns ='rank_distance')
    
    return rekomendasi_vehicle_top3

def calculate_distance(x , y, speed_x, speed_y):
        """ Kalkulasi distance euclidean dengan ratio 1.3
        Args:
                x : [latitude, longitude]
                y : [latitude, longitude]
                speed_x : float speed x
                speed_y : float speed y

        Returns:
                distance_m : distance x ke y dalam satuan meter
                duration_minutes : duration x ke y dalam satuan menit

        """
        # Menggunakan formula haversine distance untuk sphere
        lat1 = radians(x[0])
        lon1 = radians(x[1])
        lat2 = radians(y[0])
        lon2 = radians(y[1])
        R = 6373.0
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance_km = R * c 
        distance_m = distance_km*1000
        # 1.3 adalah rasio euclidean:openstreetmap
        distance_m = round(distance_m * 1.3)
        # Speed menggunakan speed rata2 dari asal ke tujuan
        speed = (speed_x + speed_y)/2
        duration_minutes = round(distance_m/speed)

        return distance_m, duration_minutes

def get_order_routing_notrouting(vehicle_idle, order, transporter_route):
    """ Memisahkan order yang dapat di-routing dan tidak dapat di routing
            Args:
                    vehicle_idle : pandas dataframe vehicle idle
                    order : pandas dataframe order unplanned
                    transporter_route : pandas dataframe rute transporter

            Returns:
                    vehicle_idle : pandas dataframe vehicle idle (preprocessed)
                    list_jenis_available : list jenis vehicle available
                    list_jenis_order : list jenis vehicle yang dibutuhkan
                    order_routing : pandas dataframe order yang jenis kendaraannya tersedia
                    order_notrouting : pandas dataframe order yang jenis kendaraannya tidak tersedia
    """
    # Get list apa saja jenis vehicle yang tersedia
    list_jenis_available = vehicle_idle['vehicle_type'].unique().tolist()
    
    # Cek apakah ada kendaraan dengan rute tersedia
    transporter_notlist = vehicle_idle[~(vehicle_idle['nama_transporter'].isin(transporter_route['nama_transporter']))].reset_index(drop= True)
    transporter_listed = transporter_route[transporter_route['nama_transporter'].isin(vehicle_idle['nama_transporter'])].reset_index(drop= True)
    list_vehicle =pd.DataFrame()
    order_routing = pd.DataFrame(columns=order.columns)
    order_notrouting = pd.DataFrame(columns=order.columns)

    for index, row in order.iterrows():
        vehicle_type = row['jenis_kendaraan']
        constraint_tahun = int(row['constraint_tahun'])
        # vehicle_route = row['vehicle_route']
        kota_asal = row['kota_asal']
        kota_tujuan = row['kota_tujuan']
        if constraint_tahun == 0 :
            temp_vehicle = vehicle_idle[vehicle_idle['vehicle_type'] == vehicle_type]
            temp_transporter = transporter_listed[(transporter_listed['nama_transporter'].isin(temp_vehicle['nama_transporter'])) & (transporter_listed['asal_kota'] == kota_asal) & (transporter_listed['tujuan_kota'] == kota_tujuan) & (transporter_listed['jenis_kendaraan'] == vehicle_type)]
            temp_transporter_nlist = transporter_notlist[transporter_notlist['vehicle_type'] == vehicle_type]
            temp_vehicle2 = temp_vehicle[temp_vehicle['nama_transporter'].isin(temp_transporter['nama_transporter'])]
            temp_list_vehicle = pd.concat([temp_vehicle2, temp_transporter_nlist], axis=0).reset_index(drop=True)
            if len(temp_vehicle) > 0 and (len(temp_transporter) > 0 or len(temp_transporter_nlist) > 0):
                order_routing.loc[len(order_routing)] = order.loc[index]
                temp_list_vehicle['id_order'] = row['id_order']
                list_vehicle = pd.concat([list_vehicle, temp_list_vehicle], axis=0).reset_index(drop = True)
            else :
                order_notrouting.loc[len(order_notrouting)] = order.loc[index]
        elif constraint_tahun > 0 :
            temp_vehicle = vehicle_idle[(vehicle_idle['vehicle_type'] == vehicle_type) & (vehicle_idle['tahun_kendaraan'] >= constraint_tahun)]
            temp_transporter = transporter_listed[(transporter_listed['nama_transporter'].isin(temp_vehicle['nama_transporter'])) & (transporter_listed['asal_kota'] == kota_asal) & (transporter_listed['tujuan_kota'] == kota_tujuan) & (transporter_listed['jenis_kendaraan'] == vehicle_type)]
            temp_transporter_nlist = transporter_notlist[(transporter_notlist['vehicle_type'] == vehicle_type) & (transporter_notlist['tahun_kendaraan'] >= constraint_tahun)]
            temp_vehicle2 = temp_vehicle[temp_vehicle['nama_transporter'].isin(temp_transporter['nama_transporter'])]
            temp_list_vehicle = pd.concat([temp_vehicle2, temp_transporter_nlist], axis=0).reset_index(drop=True)
            if len(temp_vehicle) > 0 and (len(temp_transporter) > 0 or len(temp_transporter_nlist) > 0):
                order_routing.loc[len(order_routing)] = order.loc[index]
                temp_list_vehicle['id_order'] = row['id_order']
                list_vehicle = pd.concat([list_vehicle, temp_list_vehicle], axis=0).reset_index(drop = True)
            else :
                order_notrouting.loc[len(order_notrouting)] = order.loc[index]
        else :
            order_notrouting.loc[len(order_notrouting)] = order.loc[index]
    # Bagi orderan menjadi 2 (jenis vehicle nya tersedia dan jenis vehicle nya tidak tersedia)
#     order_routing = order[order['jenis_kendaraan'].isin(list_jenis_available)].reset_index(drop = True)
#     order_notrouting = order[~(order['jenis_kendaraan'].isin(list_jenis_available))].reset_index(drop = True)
    list_jenis_order = order_routing['jenis_kendaraan'].unique().tolist()
    return vehicle_idle, list_jenis_available, list_jenis_order, order_routing, order_notrouting, list_vehicle


def get_assignment(order_routing, vehicle_idle, vehicle_type):
   """ Memisahkan order yang dapat di-routing dan tidak dapat di routing
            Args:
                    order_routing : pandas dataframe order yang dapat di-routing
                    vehicle_idle : pandas dataframe vehicle_idle
                    vehicle_type : string vehicle type

            Returns:
                    order_filtered : pandas dataframe order yang di-filter sesuai tipe kendaraan
                    dict_assignment : dict pembagian assignment berdasarkan tanggal penjemputan
                    list_assignment_vehicle : list assignment yang sudah di encoding
                    vehicle_idle : pandas dataframe vehicle idle sesuai tipe kendaraan
    """
   order_filtered = order_routing[order_routing['jenis_kendaraan'] == vehicle_type].reset_index(drop = True)
   # Buat unique_id untuk task assignment
   order_filtered['jenis_jadwal'] = order_filtered['jadwal_penjemputan_cleaned'].astype(str) + '_' + order_filtered['jenis_kendaraan']
   order_filtered[['jenis_jadwal_enc']] = order_filtered[['jenis_jadwal']].apply(LabelEncoder().fit_transform)
   order_filtered = order_filtered.drop_duplicates(subset= 'id_order')
   order_filtered = order_filtered.dropna(subset = ['id_order', 'asal', 'latitude_asal', 'longitude_asal', 'tujuan',
      'latitude_tujuan', 'longitude_tujuan', 'jadwal_penjemputan',
      'tanggal_booking', 'jenis_kendaraan', 'kota_asal', 'kota_tujuan',
   'mapping_asal', 'mapping_tujuan',
      'jadwal_penjemputan_cleaned', 'vehicle_type', 'speed(km/hour)',
      'speed_asal', 'speed_tujuan']).reset_index(drop = True)
   dict_assignment = order_filtered[['jenis_jadwal','jenis_jadwal_enc']].drop_duplicates().reset_index(drop = True).set_index('jenis_jadwal_enc').to_dict()['jenis_jadwal']


   # Karena ortools perlu memisahkan node asal dan tujuan maka perlu membuat identifier id_order - asal/ id_order - tujuan
   order_filtered['id_order_asal'] = order_filtered['id_order'] + '_'+ order_filtered['asal'].str.replace(' ', '').str.lower().str.replace('[^\w\s]','')
   order_filtered['id_order_tujuan'] = order_filtered['id_order'] + '_'+ order_filtered['tujuan'].str.replace(' ', '').str.lower().str.replace('[^\w\s]','')
   list_assignment_vehicle = order_filtered['jenis_jadwal_enc'].unique().tolist()
   vehicle_idle = vehicle_idle[vehicle_idle['vehicle_type'] == vehicle_type].reset_index(drop = True)
   
   return order_filtered, dict_assignment, list_assignment_vehicle, vehicle_idle


def get_master_node(order_filtered,leadtime_kendaraan ,vehicle_idle):
    """ Membuat tabel master node beserta variabel yang ada di node
                Args:
                        order_filtered : pandas dataframe order sudah di filter sesuai tipe kendaraan
                        leadtime_kendaraan : pandas dataframe leadtime kendaraan dari tim IT
                        vehicle_idle : pandas dataframe vehicle idle

                Returns:
                        master_routing : pandas dataframe master node
                        dummy_depot : int nomor node
                        vehicle_idle_depot : pandas dataframe vehicle with depot
                        depot_routing : pandas dataframe list depot
                        mapping_task : pandas dataframe mapping order-unique task
        """

    # Memisahkan Asal-Tujuan menjadi node terpisah
    order_asal = order_filtered[['id_order_asal', 'id_order']].rename(columns= {'id_order_asal' : 'unique_task'})
    order_tujuan = order_filtered[['id_order_tujuan', 'id_order']].rename(columns= {'id_order_tujuan' : 'unique_task'})
    mapping_task = pd.concat([order_asal, order_tujuan], axis = 0).reset_index(drop = True)
    # Melakukan encoding pada asal-tujuan menjadi unique id
    mapping_task[['encode_unique_task']] = mapping_task[['unique_task']].apply(LabelEncoder().fit_transform)
    # Menyimpan hasil encoding pada dictionary
    map_task = mapping_task.set_index('unique_task').to_dict()['encode_unique_task']

    # Membuat master node asal
    asal_routing = order_filtered[['id_order_asal', 'asal','kota_asal' ,'latitude_asal', 'longitude_asal', 'jadwal_penjemputan', 'jenis_kendaraan', 'speed_asal', 'estimasi_selesai', 'constraint_tahun']].rename(columns = {'kota_asal' : 'kota', 'id_order_asal' : 'unique_task', 'asal' : 'location', 'longitude_asal' : 'longitude', 'latitude_asal' : 'latitude', 'jadwal_penjemputan': 'jadwal', 'speed_asal' : 'speed'})
    asal_routing[['id_order','kota_asal', 'kota_tujuan']] = order_filtered[['id_order','kota_asal', 'kota_tujuan']]
    # Mapping dengan identifier
    asal_routing['unique_task'] = asal_routing['unique_task'].map(map_task)
    asal_routing['from'] = np.nan # Karena asal adalah titik jemput maka untuk kolom from akan dikosongkan
    # Membuat marker agar mudah dikenali
    asal_routing['marker'] = 'asal'
    # Estimasi loading menggunakan standard IT karena hasil ML memiliki error 22 menit
    asal_routing = pd.merge(asal_routing ,leadtime_kendaraan, how = 'left', on='jenis_kendaraan').rename(columns = {'loading_estimasi' : 'loading_duration'})
    asal_routing = asal_routing[['unique_task', 'location','kota' ,'latitude', 'longitude', 'jadwal',
        'jenis_kendaraan', 'from', 'marker', 'loading_duration', 'speed', 'estimasi_selesai', 'constraint_tahun','id_order' ,'kota_asal', 'kota_tujuan']]

    # Membuat master node Tujuan
    tujuan_routing = order_filtered[['id_order_tujuan', 'tujuan','kota_tujuan' ,'latitude_tujuan', 'longitude_tujuan','jenis_kendaraan' ,'id_order_asal', 'speed_tujuan', 'estimasi_selesai', 'constraint_tahun']].rename(columns = {'kota_tujuan' : 'kota', 'id_order_tujuan' : 'unique_task', 'id_order_asal' : 'from', 'tujuan' : 'location', 'longitude_tujuan' : 'longitude', 'latitude_tujuan' : 'latitude', 'speed_tujuan' : 'speed'})
    tujuan_routing[['id_order', 'kota_asal', 'kota_tujuan']] = order_filtered[['id_order','kota_asal', 'kota_tujuan']]
    # Mapping dengan identifier
    tujuan_routing['unique_task'] = tujuan_routing['unique_task'].map(map_task)
    tujuan_routing['from'] = tujuan_routing['from'].map(map_task)
    # Karena jadwal hanya di asal maka untuk ditujuan akan di fill dengan na
    tujuan_routing['jadwal'] = np.nan
    # Membuat marker agar mudah dikenali
    tujuan_routing['marker'] = 'tujuan'

    # Estimasi loading menggunakan standard IT karena hasil ML memiliki error 22 menit
    tujuan_routing = pd.merge(tujuan_routing ,leadtime_kendaraan, how = 'left', on='jenis_kendaraan').rename(columns = {'loading_estimasi' :'loading_duration'})
    tujuan_routing = tujuan_routing[asal_routing.columns]
    # Waktu POD di fixkan di 20 Menit (tengah2 antara 15-30 menit)
    # tujuan_routing['loading_duration'] = tujuan_routing['loading_duration'] + 20

    # Membuat master node depot
    temp_vehicle_idle = vehicle_idle[['location','kota' ,'latitude_last', 'longitude_last', 'speed_last']].drop_duplicates().reset_index(drop = True)
    # Perlu membuat dummy depot sebagai end dari trip. untuk jaraknya akan di set 0 semua sehingga dummy node ini tidak berpengaruh
    temp_vehicle_idle.loc[len(temp_vehicle_idle)] = ['dummy_depot', np.nan, 0, 0, 667]
    # Karena node depot belum memiliki unique id maka, akan dibuatkan unique_id berdasarkan no terakhir encoding
    max_mapping = mapping_task['encode_unique_task'].max() + 1
    range_mappingdepot = max_mapping + len(temp_vehicle_idle)
    unique_taskdepot = []
    for task in range(max_mapping, range_mappingdepot):
        unique_taskdepot.append(task)

    # Menambahkan variable untuk node depot
    depot_routing = temp_vehicle_idle.copy()
    depot_routing['unique_task'] = unique_taskdepot
    depot_routing = depot_routing.rename(columns = {'last_position' : 'location', 'latitude_last' : 'latitude', 'longitude_last' : 'longitude', 'speed_last' : 'speed'})
    # Karena pool tidak memiliki jadwal maka diisi np.nan
    depot_routing['jadwal'] = np.nan
    # Karena pool tidak memiliki precedence maka diisi np.nan
    depot_routing['from'] = np.nan
    # Pool tidak memiliki jenis kendaraan
    depot_routing['jenis_kendaraan'] = np.nan
    # Setting marker untuk memudahkan pengenalan
    depot_routing['marker'] = 'depot'
    depot_routing['estimasi_selesai'] = np.nan
    # Pool tidak memiliki loading duration
    depot_routing['loading_duration'] = 0
    depot_routing['constraint_tahun'] = 0
    depot_routing['id_order'] = np.nan
    depot_routing['kota_asal'] = np.nan
    depot_routing['kota_tujuan'] = np.nan
    depot_routing = depot_routing[asal_routing.columns]
    dummy_depot = depot_routing[depot_routing['location'] == 'dummy_depot']['unique_task'].values[0]

    # Concat master node asal-tujuan-pool
    master_routing = pd.concat([asal_routing, tujuan_routing, depot_routing], axis = 0).reset_index(drop = True)
    master_routing['jadwal'] = pd.to_datetime(master_routing['jadwal'])
    vehicle_idle_depot = pd.merge(vehicle_idle, depot_routing[['location', 'unique_task']], how= 'left', on= 'location')
    vehicle_idle_depot = vehicle_idle_depot.reset_index(drop = True).reset_index().rename(columns = {'index' : 'unique_vehicle'})
    vehicle_idle_depot['dummy_depot'] = dummy_depot
    master_routing['unique_task'] = master_routing['unique_task'].astype(int)
    master_routing = master_routing.sort_values(by = 'unique_task').reset_index(drop = True)
    return master_routing, dummy_depot, vehicle_idle_depot, depot_routing, mapping_task

def datetime_to_minutes(tanggal_penjemputan, tanggal_berangkat):
    """ Konversi selisih tanggal_berangkat 
        Args:
                tanggal_penjemputan : datetime tanggal penjemputan
                tanggal_berangkat : datetime tanggal_berangkat

        Returns:
                minutes : selisih tanggal penjemputan dan tanggal berangkat dalam menit
    """
    timedelta = tanggal_penjemputan - tanggal_berangkat
    seconds = timedelta.total_seconds()
    minutes = seconds/60
    return minutes

def distance_callback(from_index, to_index):
    """ Return distance 2 nodes
        Args:
            from_index : node from
            to_index : node to

        Returns:
            data['distance_matrix'][from_node][to_node] : jarak node from ke node to
    """
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

def time_callback(from_index, to_index):
    """ Return distance 2 nodes
        Args:
            from_index : node from
            to_index : node to

        Returns:
            data['time_matrix'][from_node][to_node] + data['loading_duration'][from_node] : waktu node from ke node to
    """
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['time_matrix'][from_node][to_node] + data['loading_duration'][from_node]

def print_solution(data, manager, routing, solution, temp_schedule, schedule):
    """ Get solution from ortools
        Args:
            data : dict data ortools
            manager : manager ortools
            routing : routing ortools
            solution : solution ortools
            temp_schedule : blank pandas dataframe untuk store jadwal per vehicle
            schedule : blank pandas dataframe untuk store jadwal all vehicle

        Returns:
            schedule : pandas dataframe schedule
    """

    # Routing untuk setiap vehicle
    for vehicle_id in range(data['num_vehicles']):
        # Cek start nya di index mana
        index = routing.Start(vehicle_id)
        route_order = 0
        route_distance = 0
        route_time = 0
        time_dimension = routing.GetDimensionOrDie('Time')
        # Akan terus looping selama rute nya blm berakhir
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            route_order += 1 # Route order = urutan rute yang menunjukkan harus jalan dr mana kemudian ke mana
            node = manager.IndexToNode(index) # Cek no node berdasarkan index
            next_index = solution.Value(routing.NextVar(index)) # Cek next index
            next_node = manager.IndexToNode(next_index) # Cek next node berdasarkan index
            route_distance += data['distance_matrix'][node][next_node] # Get distance dr node ke next node
            route_time += data['time_matrix'][node][next_node] + data['loading_duration'][next_node] # Get time dengan rumus = perjalanan + loading
            temp_schedule.loc[temp_schedule['unique_task'] == node,'vehicle'] = vehicle_id # Alokasikan order ke vehicle 
            temp_schedule.loc[temp_schedule['unique_task'] == node,'route_order'] = route_order # Alokasikan urutan routing ke order
            temp_schedule.loc[temp_schedule['unique_task'] == next_node,'point_distance'] = data['distance_matrix'][node][next_node] # Point distance adalah jarak dari poin sekarang ke next point
            temp_schedule.loc[temp_schedule['unique_task'] == next_node,'point_timer'] = data['time_matrix'][node][next_node] # Point timer adalah waktu dari poin sekarang ke next point
            temp_schedule.loc[temp_schedule['unique_task'] == next_node,'service_time'] = data['loading_duration'][next_node] # Service time adalah waktu loading/unloading pada node tujuan
            temp_schedule.loc[temp_schedule['unique_task'] == node,'end_time_min'] = solution.Min(time_var) # Time minimal harus dilokasi
            temp_schedule.loc[temp_schedule['unique_task'] == node,'end_time_max'] = solution.Max(time_var) # Time maksimal harus dilokasi
            previous_index = index # Assign index sebelumnya ke variabel
            index = solution.Value(routing.NextVar(index)) # Get next index
        route_order += 1 # Route order terakhir
        node = manager.IndexToNode(index) # Cek no node berdasarkan index
        temp_schedule.loc[temp_schedule['unique_task'] == node,'vehicle'] = vehicle_id # Alokasikan node ke vehicle
        temp_schedule.loc[temp_schedule['unique_task'] == node,'route_order'] = route_order # Alokasikan urutan order ke node
        temp_schedule.loc[temp_schedule['vehicle'] == vehicle_id,'route_distance'] = route_distance # Route Distance adalah total distance dari 1 rute perjalanan
        temp_schedule.loc[temp_schedule['vehicle'] == vehicle_id,'route_time'] = route_time # Route time adalah total time dari 1 rute perjalanan
        temp_schedule.loc[temp_schedule['unique_task'] == node,'end_time_min'] = solution.Min(time_var) # Time minimal harus dilokasi
        temp_schedule.loc[temp_schedule['unique_task'] == node,'end_time_max'] = solution.Max(time_var) # Time maksimal harus dilokasi
        schedule = pd.concat([schedule, temp_schedule.loc[temp_schedule['vehicle'] == vehicle_id]], axis=0 ).reset_index(drop = True) # Concat temp_schedule final ke dataframe kosong
    return schedule

def prepare_variable(master_routing, dummy_depot, vehicle_idle_depot, list_vehicle_combination):
    """ Get solution from ortools
        Args:
            master_routing : pandas dataframe master routing node
            dummy_depot : int no depot
            vehicle_idle_depot : pandas dataframe vehicle idle dan informasi depot
            vehicle_combination : pandas dataframe kombinasi jadwal vehicle terhadap order
        Returns:
            distance_matrix : distance matrix
            duration_matrix : duration matrix
            time_window : pandas dataframe time window setiap node
            del_drop : pandas dataframe setting delivery dan drop
            jadwal_berangkat : datetime jadwal berangkat
    """
    # Membuat matrix Distance & matrix Duration
    latlon_coordinate = master_routing[['unique_task', 'latitude', 'longitude', 'speed']].sort_values(by = 'unique_task')
    distance_matrix = np.zeros([len(latlon_coordinate), len(latlon_coordinate)]) 
    duration_matrix = np.zeros([len(latlon_coordinate), len(latlon_coordinate)])
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
                koordinat_asal = [latlon_coordinate['latitude'][i],  latlon_coordinate['longitude'][i]]
                koordinat_tujuan = [latlon_coordinate['latitude'][j],  latlon_coordinate['longitude'][j]]
                speed_x = latlon_coordinate['speed'][i]
                speed_y = latlon_coordinate['speed'][j]
                distance_m, duration_minutes = calculate_distance(koordinat_asal, koordinat_tujuan, speed_x, speed_y)
                distance_matrix[i,j] = distance_m
                duration_matrix[i,j] = duration_minutes

    # Matrix Distance perlu ditambahkan depot dummy
    distance_matrix = pd.DataFrame(distance_matrix,index=latlon_coordinate['unique_task'],
            columns=latlon_coordinate['unique_task'])
    distance_matrix = distance_matrix.reset_index().drop(columns = 'unique_task')
    distance_matrix.columns.name = None
    distance_matrix.loc[:,dummy_depot] = 0
    distance_matrix.loc[dummy_depot,:] = 0
    
    # Matrix Duration perlu ditambahkan depot dummy
    duration_matrix = pd.DataFrame(duration_matrix,index=latlon_coordinate['unique_task'],
            columns=latlon_coordinate['unique_task'])
    duration_matrix = duration_matrix.reset_index().drop(columns = 'unique_task')
    duration_matrix.columns.name = None
    duration_matrix.loc[:,dummy_depot] = 0
    duration_matrix.loc[dummy_depot,:] = 0
   
    # Karena tujuan dan asal pada order di-split menjadi 2, maka perlu di-definisikan 1 set antara asal ke tujuan
    del_drop = master_routing[['unique_task', 'from']].dropna().reset_index(drop= True)
    del_drop['from'] = del_drop['from'].astype(int)

    # Time Window adalah waktu yang diizinkan untuk sampai ke node
    minutes_berangkat = round(duration_matrix.max().max()) # Perlu set jadwal berangkat. Untuk jadwal berangkat menggunakan rumus tanggal order - maksimal keberangkatan
    jadwal_min = master_routing['jadwal'].min()
    jadwal_berangkat = jadwal_min - timedelta(minutes=minutes_berangkat)
    # Perlu get index mana saja yang termasuk node asal untuk melakukan apply function dan mendapatkan jadwal berangkat
    index_asal = master_routing[master_routing['marker'] == 'asal'].index
    # Update multi-pick & multi-drop sesuai dengan estimasi selesai
    order_multi = master_routing.dropna(subset='estimasi_selesai').dropna(subset='from').reset_index(drop= True)
    if len(order_multi) > 0 :
        for index, row in order_multi.iterrows():
                asal = int(row['from'])
                tujuan = int(row['unique_task'])
                estimasi_selesai = row['estimasi_selesai']
                minutes_update = int((estimasi_selesai - jadwal_berangkat).seconds/60)
                duration_matrix.loc[asal,tujuan] = minutes_update
                duration_matrix.loc[tujuan,asal] = minutes_update

    # Apply function untuk mendapatkan jadwal berangkat
    master_routing.loc[index_asal,'end_timewindow'] = master_routing.loc[index_asal,'jadwal'].apply(lambda x: datetime_to_minutes(x ,jadwal_berangkat))  # Time window strict start & end tidak ada bedanya
    master_routing.loc[index_asal,'start_timewindow'] = master_routing.loc[index_asal,'jadwal'].apply(lambda x: datetime_to_minutes(x ,jadwal_berangkat))
    master_routing['start_timewindow'] = master_routing['start_timewindow'].apply(lambda x : x if x > 0 else 0) # yg null akan diisi 0
    time_window = master_routing[['unique_task','start_timewindow', 'end_timewindow' ]].dropna().reset_index(drop= True) # drop yang NA
    time_window['start_timewindow'] = time_window['start_timewindow'].astype(int) # krn ortools harus integer maka perlu diconvert ke int
    time_window['end_timewindow'] = time_window['end_timewindow'].astype(int)  # krn ortools harus integer maka perlu diconvert ke int
    # krn ortools harus integer maka perlu diconvert ke int
    distance_matrix = distance_matrix.astype(int) 
    duration_matrix = duration_matrix.astype(int) 
    time_window = time_window.astype(int) 
    del_drop = del_drop.astype(int)  
    vehicle_idle_depot['unique_task'] = vehicle_idle_depot['unique_task'].astype(int)
    vehicle_idle_depot['dummy_depot'] = vehicle_idle_depot['dummy_depot'].astype(int)
    master_routing['loading_duration'] = master_routing['loading_duration'].astype(int)

    # Restriction vehicle
    vehicle_combination = list_vehicle_combination[['nopol','id_order']].merge(vehicle_idle_depot[['unique_vehicle', 'nopol']], how= 'left', on = 'nopol')
    node_order = master_routing[['unique_task', 'id_order']].dropna()
    vehicle_combination = pd.merge(node_order, vehicle_combination, on='id_order')
    vehicle_combination['unique_vehicle'] = vehicle_combination['unique_vehicle'].astype(int)

    return distance_matrix, duration_matrix, time_window, del_drop, jadwal_berangkat, vehicle_combination

def rules_pool(schedule, vehicle_idle_depot, mapping_task, berangkat, list_order_assign, list_nopol_assign):
    """ Menerapkan rules pool
        Args:
            schedule : pandas dataframe master routing node
            vehicle_idle_depot : int no depot
            mapping_task : pandas dataframe vehicle idle dan informasi depot
        Returns:
            schedule : pandas dataframe schedule
            list_order_assign : list order yang di-assign
            list_nopol_assign : list nopol yang di-assign
    """

    schedule = schedule.sort_values(by=['vehicle', 'route_order']).reset_index(drop = True) # Sort value berdasarkan vehicle dan route order
    schedule = schedule.groupby('vehicle').filter(lambda x: len(x) > 2).reset_index(drop = True) # jika len nya hanya 2 berarti hanya start dan end saja (tidak mengunjungi node sama sekali). Oleh karena itu perlu ditake-out
    schedule = schedule[schedule['location'] != 'dummy_depot'].reset_index(drop = True) # Dummy depot tidak disertakan dalam rules
    vehicle_join = vehicle_idle_depot.rename(columns= {'unique_vehicle' : 'vehicle', 'nama_pool' : 'location_depot', 'latitude_pool': 'latitude_depot', 'longitude_pool' : 'longitude_depot'}).drop(columns = {'unique_task','location', 'kota' })
    schedule = pd.merge(schedule, vehicle_join, how= 'left', on= 'vehicle') # Join jadwal dengan kendaraan
    list_vrp_vehicle = schedule['vehicle'].unique().tolist() # Akan dilakukan looping per kendaraan untuk mengecek dan menerapkan rules ke setiap kendaraan oleh karena itu dibutuhkan list kendaraan
    for vehicle in list_vrp_vehicle:
        df = schedule[schedule['vehicle'] == vehicle].sort_values(by = 'route_order')
        rules_pool = df['rules_pool'].unique().tolist()[0] # Untuk mengetahui rules pool nya apakah harus kembali atau tidak
        lokasi_pool = df['mapping_rules'].unique().tolist()[0] # Untuk mengetahui lokasi pool nya ada dimana
        if rules_pool == 'YA' and lokasi_pool == 'Jabodetabek': # Jika iya dan lokasinya di jabodetabek maka perlu cek apakah distance terakhir < 40 km?
            latlon_last = df.iloc[-1][['latitude', 'longitude']].tolist() # latlon last position
            latlon_depot = df.iloc[-1][['latitude_depot', 'longitude_depot']].tolist() # Cek latlon depot
            service_time = df.iloc[-1]['service_time'] # Service time di lokasi akhir
            speed_last = df.iloc[-1]['speed'] # Speed lokasi akhir
            speed_depot = df.iloc[-1]['speed_depot']
            distance_m, duration_minutes = calculate_distance(latlon_last,latlon_depot, speed_last, speed_depot) # perhitungan waktu dan durasi
            distance_km = distance_m/1000 # konversi distance ke KM
            if distance_km < 40 : # Jika kurang dari 40 km maka kembali ke pool
                route_order = df.iloc[-1]['route_order']
                end_time_min = df.iloc[-1]['end_time_min']
                end_time_max = df.iloc[-1]['end_time_max']
                new_route_order = route_order + 1 # karena kembali ke pool maka urutan di + 1
                new_end_min = end_time_min + duration_minutes + service_time # new time = durasi min + durasi perjalanan + durasi loading di lokasi akhir
                new_end_max = end_time_max + duration_minutes + service_time # new time = durasi max + durasi perjalanan + durasi loading di lokasi akhir
                new_route_distance = df.iloc[-1]['route_distance'] + distance_m # distance untuk 1x trip. perlu di + sama distance ke pool
                new_route_time = df.iloc[-1]['route_time'] + duration_minutes + service_time # time untuk 1x trip. perlu di + sama time ke pool dan service time
                new_route = {
                'unique_task': np.nan, 
                'location' : df.iloc[-1]['location_depot'], 
                'latitude': df.iloc[-1]['latitude_depot'], 
                'longitude': df.iloc[-1]['longitude_depot'], 
                'jadwal': np.nan,
                'jenis_kendaraan':df.iloc[-1]['jenis_kendaraan'] , 
                'from': np.nan, 
                'rank' : np.nan,
                'marker': 'depot', 
                'speed' : speed_depot, # satuannya meter/minutes
                'end_timewindow': np.nan,
                'start_timewindow': np.nan, 
                'vehicle' : vehicle, 
                'route_order': new_route_order, 
                'point_distance': distance_m,
                'point_timer' : duration_minutes, 
                'service_time' : 0,
                'end_time_min' :new_end_min, 
                'end_time_max': new_end_max, 
                'route_distance': new_route_distance,
                'route_time': new_route_time, 
                'dummy_demand' : 0, 
                'loading_duration' : 0, 
                'nopol' : df.iloc[-1]['nopol'],
                'vehicle_type': df.iloc[-1]['vehicle_type'], 
                'location_depot' : df.iloc[-1]['location_depot'], 
                'mapping_lokasi' : df.iloc[-1]['mapping_lokasi'],
                'mapping_rules' : df.iloc[-1]['mapping_rules'],
                'rules_pool' : df.iloc[-1]['rules_pool'], 
                'latitude_depot': df.iloc[-1]['latitude_depot'], 
                'longitude_depot' : df.iloc[-1]['longitude_depot']
                }
                schedule = schedule.append(new_route, ignore_index = True)
                schedule.loc[schedule['vehicle'] == vehicle,'route_distance'] = new_route_distance # Perbarui distance rute
                schedule.loc[schedule['vehicle'] == vehicle,'route_time'] = new_route_time # Perbarui time rute
            else:
                # Kalau di atas 40 km berarti tetap di lokasi akhir
                latlon_last = df.iloc[-1][['latitude', 'longitude']].tolist()
                service_time = df.iloc[-1]['service_time'] 
                route_order = df.iloc[-1]['route_order']
                end_time_min = df.iloc[-1]['end_time_min']
                end_time_max = df.iloc[-1]['end_time_max']
                speed_last = 0 # nggak ngaruh karena kalau tetap di lokasi akhir artinya tidak berpindah
                new_route_order = route_order + 1
                new_end_min = end_time_min + service_time
                new_end_max = end_time_max + service_time
                new_route_distance = df.iloc[-1]['route_distance']
                new_route_time = df.iloc[-1]['route_time'] + service_time
                new_route = {
                'unique_task': np.nan, 
                'location' : str(df.iloc[-1]['location']) + '- Standby', 
                'latitude': df.iloc[-1]['latitude'], 
                'longitude': df.iloc[-1]['longitude'], 
                'jadwal': np.nan,
                'jenis_kendaraan':df.iloc[-1]['jenis_kendaraan'] , 
                'from': np.nan, 
                'rank' : np.nan,
                'marker': 'depot', 
                'speed' : 0,
                'end_timewindow': np.nan,
                'start_timewindow': np.nan, 
                'vehicle' : vehicle, 
                'route_order': new_route_order, 
                'point_distance': 0,
                'point_timer' : 0, 
                'service_time' : 0,
                'end_time_min' :new_end_min, 
                'end_time_max': new_end_max, 
                'route_distance': new_route_distance,
                'route_time': new_route_time, 
                'dummy_demand' : 0, 
                'loading_duration' : 0, 
                'nopol' : df.iloc[-1]['nopol'],
                'vehicle_type': df.iloc[-1]['vehicle_type'], 
                'location_depot' : df.iloc[-1]['location_depot'], 
                'mapping_lokasi' : df.iloc[-1]['mapping_lokasi'],
                'mapping_rules' : df.iloc[-1]['mapping_rules'],
                'rules_pool' : df.iloc[-1]['rules_pool'], 
                'latitude_depot': df.iloc[-1]['latitude_depot'], 
                'longitude_depot' : df.iloc[-1]['longitude_depot']
                }
                schedule = schedule.append(new_route, ignore_index = True)
                schedule.loc[schedule['vehicle'] == vehicle,'route_distance'] = new_route_distance
                schedule.loc[schedule['vehicle'] == vehicle,'route_time'] = new_route_time
        else:
            # Kalau tidak berarti auto kembali ke pool
            latlon_last = df.iloc[-1][['latitude', 'longitude']].tolist()
            latlon_depot = df.iloc[-1][['latitude_depot', 'longitude_depot']].tolist()
            service_time = df.iloc[-1]['service_time'] 
            speed_last = df.iloc[-1]['speed'] 
            speed_depot = df.iloc[-1]['speed_depot']
            distance_m, duration_minutes = calculate_distance(latlon_last,latlon_depot,speed_last, speed_depot ) 
            distance_km = distance_m/1000 # Konversi distance ke km
            route_order = df.iloc[-1]['route_order']
            end_time_min = df.iloc[-1]['end_time_min']
            end_time_max = df.iloc[-1]['end_time_max']
            new_route_order = route_order + 1
            new_end_min = end_time_min + duration_minutes + service_time
            new_end_max = end_time_max + duration_minutes + service_time
            new_route_distance = df.iloc[-1]['route_distance'] + distance_m
            new_route_time = df.iloc[-1]['route_time'] + duration_minutes + service_time
            new_route = {
            'unique_task': np.nan, 
            'location' : df.iloc[-1]['location_depot'], 
            'latitude': df.iloc[-1]['latitude_depot'], 
            'longitude': df.iloc[-1]['longitude_depot'], 
            'jadwal': np.nan,
            'jenis_kendaraan':df.iloc[-1]['jenis_kendaraan'] , 
            'from': np.nan, 
            'rank' : np.nan,
            'marker': 'depot', 
            'speed' : speed_depot, # satuan speed meter/menit
            'end_timewindow': np.nan,
            'start_timewindow': np.nan, 
            'vehicle' : vehicle, 
            'route_order': new_route_order, 
            'point_distance': distance_m,
            'point_timer' : duration_minutes, 
            'service_time' : 0, # gaada kegiatan loading & unloading
            'end_time_min' :new_end_min, 
            'end_time_max': new_end_max, 
            'route_distance': new_route_distance,
            'route_time': new_route_time, 
            'dummy_demand' : 0, 
            'loading_duration' : 0, 
            'nopol' : df.iloc[-1]['nopol'],
            'vehicle_type': df.iloc[-1]['vehicle_type'], 
            'location_depot' : df.iloc[-1]['location_depot'], 
            'mapping_lokasi' : df.iloc[-1]['mapping_lokasi'],
            'mapping_rules' : df.iloc[-1]['mapping_rules'],
            'rules_pool' : df.iloc[-1]['rules_pool'], 
            'latitude_depot': df.iloc[-1]['latitude_depot'], 
            'longitude_depot' : df.iloc[-1]['longitude_depot']
            }
            schedule = schedule.append(new_route, ignore_index = True)
            schedule.loc[schedule['vehicle'] == vehicle,'route_distance'] = new_route_distance
            schedule.loc[schedule['vehicle'] == vehicle,'route_time'] = new_route_time
    schedule = schedule.sort_values(by=['vehicle', 'route_order']).reset_index(drop = True) # Sort values by vehicle & urutan rute trip
    schedule['end_time_min_conv'] = schedule['end_time_min'].apply(lambda x: berangkat + timedelta(minutes = x)) # konversi waktu ke satuan datetime
    schedule['end_time_max_conv'] = schedule['end_time_max'].apply(lambda x: berangkat + timedelta(minutes = x)) # konversi waktu ke satuan datetime
    schedule['point_distance_km'] = (schedule['point_distance']/1000).round(2) # Ubah ke satuan km
    schedule['point_timer_hour'] = (schedule['point_timer']/60).round(2) # Ubah ke satuan jam
    schedule['route_distance_km'] = (schedule['route_distance']/1000).round(2) # Ubah ke satuan km
    schedule['route_time_hour'] = (schedule['route_time']/60).round(2) # Ubah ke satuan jam
    schedule['service_time_hour'] = (schedule['service_time']/60).round(2) # Ubah ke satuan jam
    schedule['loading_duration_hour'] = (schedule['loading_duration']/60).round(2) # Ubah ke satuan jam
    mapping_taskmerge = mapping_task[['encode_unique_task', 'id_order']].rename(columns= {'encode_unique_task' : 'unique_task'}) # ambil id unique-id order
    schedule = pd.merge(schedule, mapping_taskmerge, how= 'left', on= 'unique_task') # Join untuk return ke id order
    
    order_notnull = schedule[schedule['marker'] != 'depot']
    list_order_assign.extend(order_notnull['id_order'].unique().tolist())
    list_nopol_assign.extend(schedule['nopol'].unique().tolist())

    
    return schedule, list_order_assign, list_nopol_assign

def generate_report(jadwal_lengkap, order_filtered, vehicle_idle, list_order_assign, list_nopol_assign):
    """ Generate report
        Args:
            schedule : pandas dataframe master routing node
            vehicle_idle_depot : int no depot
            mapping_task : pandas dataframe vehicle idle dan informasi depot
        Returns:
            routing_detail_report : pandas dataframe detail routing
            routing_report : pandas dataframe routing by order
            trip_report : pandas dataframe routing by nopol
            vehicle_report : pandas dataframe vehicle report
            not_routing : pandas dataframe order not routing
            vehicle_sisa : pandas dataframe vehicle sisa
    """
    # Report detail routing
    routing_detail_report = jadwal_lengkap[['nopol', 'vehicle_type', 'id_order', 'location','kota' , 'jadwal','route_order', 'point_distance_km', 'point_timer_hour','service_time_hour','end_time_min_conv', 'end_time_max_conv' ]].rename(columns ={
    'nopol' : 'no_polisi',
    'route_order' : 'urutan_rute',
    'point_distance_km' : 'jarak_km',
    'point_timer_hour' : 'waktu_jam',
    'service_time_hour' : 'service_jam',
    'end_time_min_conv' : 'min_time_location', 
    'end_time_max_conv' : 'max_time_location'
    })
    routing_detail_report.loc[(routing_detail_report['urutan_rute'] == 1), 'start_location'] = routing_detail_report.loc[(routing_detail_report['urutan_rute'] == 1), 'max_time_location'] # Gunakan waktu max untuk keberangkatan dari depot/pool
    routing_detail_report.loc[(routing_detail_report['urutan_rute'] != 1), 'start_location'] = routing_detail_report.loc[(routing_detail_report['urutan_rute'] != 1), 'min_time_location'] # Gunakan waktu min untuk keberangkatan selain dr pool
    routing_detail_report = routing_detail_report.drop(columns=['min_time_location','max_time_location']) # Drop kolom karena sudah tidak dibutuhkan
    routing_detail_report['service_jam'] = routing_detail_report['service_jam'].fillna(0) # Fillna dengan 0
    routing_detail_report['waktu_jam'] = routing_detail_report['waktu_jam'].fillna(0) # Fillna dengan 0
    routing_detail_report['jarak_km'] = routing_detail_report['jarak_km'].fillna(0) # Fillna dengan 0
    routing_detail_report['departure_location_'] = routing_detail_report.apply(lambda x: x['start_location'] - timedelta(hours = x['waktu_jam']), axis=1) # waktu berangkat adalah start location dikurangi perjalanan
    routing_detail_report['departure_location'] = routing_detail_report.groupby('no_polisi')['departure_location_'].shift(-1) # tapi perlu diletakkan di baris sebelumnya karena waktu berangkat dr lokasi sebelumnya
    routing_detail_report = routing_detail_report.drop(columns = {'departure_location_'}) # drop kolom dummy
    routing_detail_report['end_location'] = routing_detail_report.apply(lambda x: x['start_location'] + timedelta(hours = x['service_jam']), axis=1) # waktu selesai dilokasi = waktu sampai + waktu loading/unloading
    routing_detail_report['departure_location'] = routing_detail_report['departure_location'].fillna(routing_detail_report['end_location']) # Departure yang kosong diisi dengan waktu end location
    routing_detail_report['waiting_location'] = routing_detail_report['departure_location'] - routing_detail_report['end_location'] # Slack = waktu departure - waktu selesai loading/unloading
    routing_detail_report['waiting_location'] = (routing_detail_report['waiting_location'].dt.total_seconds()/3600).round(2) # Ubah ke satuan jam
    routing_detail_report.loc[routing_detail_report['waiting_location'] <= 0, 'waiting_location'] = 0 # yang minus biasanya ada selisih de second/minutes. karena itu konversi ke 0
    routing_detail_report = routing_detail_report[['no_polisi', 'vehicle_type', 'id_order', 'location', 'kota', 'jadwal',
       'urutan_rute', 'jarak_km', 'waktu_jam', 'service_jam', 'start_location', 'end_location','departure_location',  'waiting_location']]
    routing_detail_report = routing_detail_report.drop_duplicates().reset_index(drop = True)
    
    # Report Routing per order
    routing_detail_report['id_order'] = routing_detail_report['id_order'].fillna('depot').reset_index(drop = True)
    order_notnull = routing_detail_report[routing_detail_report['id_order'] != 'depot']
    order_isnull = routing_detail_report[routing_detail_report['id_order'] == 'depot']
    # Aggregasikan untuk mendapatkan report per id order & vehicle
    ordernotnull_agg = order_notnull.groupby(['no_polisi', 'vehicle_type', 'id_order']).agg({'urutan_rute' : 'min', 'jarak_km' : 'sum', 'waktu_jam' : 'sum','service_jam' : 'sum','waiting_location' : 'sum' ,'jadwal' : 'min', 'start_location' : 'min', 'end_location' : 'max'}).reset_index()
    order_isnull =order_isnull[ordernotnull_agg.columns] # depot tidak perlu agregasi krn gapunya order
    routing_report = pd.concat([order_isnull, ordernotnull_agg], axis = 0).reset_index(drop = True)
    routing_report = routing_report.sort_values(by= ['no_polisi', 'urutan_rute']).reset_index(drop = True)
    routing_report['route_order'] = routing_report.groupby("no_polisi")["urutan_rute"].rank(method="dense", ascending=True).astype(int).reset_index(drop = True) # Calculate ulang route order
    routing_report = routing_report.drop_duplicates().reset_index(drop = True)

    # Report summary order
    trip_report = order_notnull.groupby(['id_order', 'no_polisi','vehicle_type']).agg({'jadwal' : 'min', 'start_location' : 'min', 'end_location' : 'max'}).reset_index().rename(columns = {'start_location' : 'start', 'end_location' : 'end'})
    trip_report = trip_report.sort_values(by= ['no_polisi', 'start'], ascending =[True, True]).reset_index(drop = True)
    trip_report['route_order'] = trip_report.groupby('no_polisi')['start'].rank('dense')
    trip_report = trip_report.drop_duplicates().reset_index(drop = True)
    

    # Report per Vehicle
    routing_report_dummy = routing_report.copy()
    vehicle_report = routing_report_dummy.groupby(['no_polisi', 'vehicle_type']).agg({'jarak_km' : 'sum', 'waktu_jam' : 'sum','service_jam' : 'sum','waiting_location' : 'sum','start_location' : 'min', 'end_location' : 'max' }).reset_index(drop = False)
    vehicle_report = vehicle_report.rename(columns= {'start_location' : 'departure_time', 'end_location' : 'end_time'})
    vehicle_report = vehicle_report.drop_duplicates().reset_index(drop = True)
    
    # Report order yang tidak dapat multi-trip
    not_routing = order_filtered[~order_filtered['id_order'].isin(list_order_assign)].reset_index(drop = True)
    not_routing = not_routing.drop_duplicates().reset_index(drop = True)

    # Report Sisa Vehicle yang tidak dijadwalkan
    vehicle_sisa = vehicle_idle[~vehicle_idle['nopol'].isin(list_nopol_assign)].reset_index(drop = True)
    vehicle_sisa = vehicle_sisa.drop_duplicates().reset_index(drop = True)


    return routing_detail_report, routing_report, trip_report, vehicle_report, not_routing, vehicle_sisa

header()
set_bg_hack_url()
st.subheader('**Chaining Routing Recommendation**')
st.write("**- Silahkan edit input pada [spreadsheet](https://docs.google.com/spreadsheets/d/1OE3vzXS3uVd2Y96tS165VbEBN7RVHK9EOFcc6onOGGo/edit#gid=16022336) berikut sebelum menjalankan engine**")
st.write("**- Output dari engine dapat dilihat di web atau dapat masuk ke link [spreadsheet](https://docs.google.com/spreadsheets/d/15iw6I7-qTlf1mHgHeucrPegPPwE8a5utJHg2IIFMZXg/edit#gid=1031787686) berikut**")
st.write("**- Output di spreadsheet akan otomatis update ketika engine dijalankan**")
st.markdown('**Tombol berikut berfungsi untuk menjalankan engine routing recommendation**')

if st.button('**Run Engine**') :
    # Read Google Sheet Output
    # Define Scope
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive.file','https://www.googleapis.com/auth/drive']
    # Add your service account file
    creds = ServiceAccountCredentials.from_json_keyfile_name('credential.json', scope)
    # Authorize the clientsheet 
    client = gspread.authorize(creds)
    sheet = client.open('P2_output_engine')
    # Define each Sheet
    # sheet_routing_hard_constraint = sheet.worksheet('routing_hard_constraint')
    sheet_recommendation_vehicle_hard_constraint = sheet.worksheet('recommendation_vehicle_hard_constraint')
    sheet_recommendation_vehicle_hard_constraint.clear()
    # sheet_not_feasible_hard_constraint = sheet.worksheet('not_feasible_hard_constraint')
    # sheet_remaining_vehicle_hard_constraint = sheet.worksheet('remaining_vehicle_hard_constraint')
    sheet_recommendation_vehicle = sheet.worksheet('recommendation_vehicle')
    sheet_recommendation_vehicle.clear()
    # sheet_routing_soft_constraint = sheet.worksheet('routing_soft_constraint')
    sheet_recommendation_vehicle_soft_constraint = sheet.worksheet('recommendation_vehicle_soft_constraint')
    sheet_recommendation_vehicle_soft_constraint.clear()
    # sheet_not_feasible_soft_constraint = sheet.worksheet('not_feasible_soft_constraint')
    # sheet_remaining_vehicle_soft_constraint = sheet.worksheet('remaining_vehicle_soft_constraint')
    sheet_not_routing = sheet.worksheet('not_routing')
    sheet_not_routing.clear()
    sheet_vehicle_sisa = sheet.worksheet('vehicle_sisa')
    sheet_vehicle_sisa.clear()
    start_hitung = tm.time()

    # Import Data dari spreadsheet
    # sheet_id = '1dHMcwlYez_MAthxEX52QspZ0dcI8pnoqMcIWsuYtBd4'
    sheet_id = '1OE3vzXS3uVd2Y96tS165VbEBN7RVHK9EOFcc6onOGGo'
    tab_order = 'order'
    tab_vehicle = 'idle_vehicle'
    tab_region = 'master_mapping'
    tab_leadtime = 'master_leadtime'
    tab_pool = 'master_pool'
    tab_speed = 'master_speed'
    tab_mvehicle = 'master_vehicle' # change
    tab_transporter = 'transporter_constraint' # change
    tab_mlocation = 'master_location' # change
    url_order = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_order}"
    url_vehicle = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_vehicle}"
    url_region = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_region}"
    url_leadtime = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_leadtime}"
    url_pool = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_pool}"
    url_speed = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_speed}"
    url_transporter = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_transporter}" # change
    url_mvehicle = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_mvehicle}" # change
    url_mlocation = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_mlocation}" # change

    df_vehicle_idle = pd.read_csv(url_vehicle)
    df_order = pd.read_csv(url_order)
    df_region = pd.read_csv(url_region)
    df_leadtime_kendaraan = pd.read_csv(url_leadtime)
    df_pool = pd.read_csv(url_pool)
    df_speed_region = pd.read_csv(url_speed)
    df_transporter = pd.read_csv(url_transporter) # change
    df_mvehicle = pd.read_csv(url_mvehicle) # change
    df_mlocation = pd.read_csv(url_mlocation) # change

    # Preprocessing
    df_mlocation['constraint_tahun'] = pd.to_numeric(df_mlocation['constraint_tahun'], errors='coerce', downcast='integer').fillna(0).astype(int)
    min_year = pd.to_numeric(df_mvehicle['tahun_kendaraan'], errors='coerce').dropna().min()
    # Yang gaada tahun kendaraanya di fillna pakai min year
    df_mvehicle['tahun_kendaraan'] = pd.to_numeric(df_mvehicle['tahun_kendaraan'], errors='coerce', downcast='integer').fillna(min_year).astype(int)
    df_vehicle_idle = df_vehicle_idle.dropna(subset=['nopol', 'vehicle_type','nama_transporter' ,'nama_pool', 'last_position', 'kota',
        'latitude_last', 'longitude_last']).drop_duplicates(subset = ['nopol']).reset_index(drop = True)
    df_order = df_order.dropna(subset='id_order').reset_index(drop = True)

    # Vehicle ready preprocessed
    df_vehicle_idle = pd.merge(df_vehicle_idle.drop(columns ={'nama_pool'}), df_pool, how='left', on= 'nama_transporter').merge(df_region, on= 'kota', how = 'left').rename(columns ={'mapping_region' : 'region_last'}).merge(df_speed_region, how= 'left', left_on= ['vehicle_type', 'region_last'], right_on= ['vehicle_type', 'region']).drop(columns= {'speed(km/hour)','region'}).rename(columns = {'speed(m/min)' : 'speed_last'}).merge(df_speed_region, how= 'left', left_on= ['vehicle_type', 'mapping_lokasi'], right_on= ['vehicle_type', 'region']).drop(columns= {'speed(km/hour)','region'}).rename(columns = {'speed(m/min)' : 'speed_depot'})
    df_vehicle_idle = pd.merge(df_vehicle_idle, df_mvehicle[['nopol','plan_awal','active_flag','tahun_kendaraan']], how='left', on='nopol')
    # Yang gaada tahun kendaraanya sementara di fillna pakai min year
    df_vehicle_idle['tahun_kendaraan'] = df_vehicle_idle['tahun_kendaraan'].fillna(min_year).astype(int)
    # Filter kendaraan yang gapunya plan awal & active
    df_vehicle_idle = df_vehicle_idle[(df_vehicle_idle['plan_awal'].isnull()) & (df_vehicle_idle['active_flag'] == True)].reset_index(drop = True)

    # Join order & region
    df_order = pd.merge(df_order, df_region, how = 'left', left_on = 'kota_asal', right_on = 'kota').drop(columns= 'kota').rename(columns= {'mapping_region' : 'mapping_asal'})
    df_order = pd.merge(df_order, df_region, how = 'left', left_on = 'kota_tujuan', right_on = 'kota').drop(columns= 'kota').rename(columns= {'mapping_region' : 'mapping_tujuan'})
    # Drop yang tidak ada mapping nya
    df_order = df_order.dropna(subset= ['mapping_asal', 'mapping_tujuan']).reset_index(drop = True)
    df_order = pd.merge(df_order, df_mlocation[['lokasi', 'constraint_tahun']],how= 'left' ,left_on='asal', right_on='lokasi').drop(columns='lokasi').rename(columns= {'constraint_tahun' : 'constraint_asal'}) \
    .merge(df_mlocation[['lokasi','constraint_tahun']],how= 'left' ,left_on='tujuan', right_on='lokasi').drop(columns='lokasi').rename(columns= {'constraint_tahun' : 'constraint_tujuan'})
    df_order['constraint_tahun'] = df_order[['constraint_asal', 'constraint_tujuan']].max(axis=1).fillna(0)
    list_column_date = ['jadwal_penjemputan', 'estimasi_selesai']
    df_vehicle_idle = df_vehicle_idle[['nopol', 'vehicle_type','nama_transporter','nama_pool', 'mapping_lokasi','mapping_rules' ,'rules_pool',
                                                            'latitude_pool', 'longitude_pool', 'last_position','kota' ,'latitude_last', 'longitude_last', 'speed_last',
                                                            'speed_depot','tahun_kendaraan'
                                                            ]].rename(columns = {'last_position' : 'location'})
    # Convert Data type
    for column in list_column_date:
        df_order[column] = pd.to_datetime(df_order[column])
    df_order['jadwal_penjemputan_cleaned'] = df_order['jadwal_penjemputan'].dt.strftime('%d/%m/%Y')
    df_order = pd.merge(df_order, df_speed_region, how= 'left', left_on = ['mapping_asal', 'jenis_kendaraan'], right_on=['region', 'vehicle_type'],  suffixes=('', '_remove')).drop(columns={'region'}).rename(columns= {'speed(m/min)' : 'speed_asal'}).merge(df_speed_region, how= 'left', left_on = ['mapping_tujuan', 'jenis_kendaraan'], right_on=['region', 'vehicle_type'],  suffixes=('', '_remove')).drop(columns={'region'}).rename(columns= {'speed(m/min)' : 'speed_tujuan'})
    df_order.drop([i for i in df_order.columns if 'remove' in i],
                axis=1, inplace=True) # Drop duplicate columns

    st.subheader('**Data Vehicle Idle**')
    st.markdown('**Data vehicle idle adalah data vehicle yang dapat digunakan untuk assignment. Data vehicle idle mencakup informasi vendor dan posisi terakhir vehicle.**')
    df_vehicle_idle
    st.subheader('**Data Order**')
    st.markdown('**Data order adalah data order yang belum di-planning.**')
    df_order
    st.subheader('**Data Region**')
    st.markdown('**Data mapping region berisi data mapping kota berdasarkan region nya. Data ini digunakan untuk penentuan speed dari vehicle.**')
    df_region
    st.subheader('**Data Leadtime Kendaraan**')
    st.markdown('**Data ini berisi leadtime untuk kegiatan loading dan unloading pada setiap kendaraan.**')
    df_leadtime_kendaraan
    st.subheader('**Data Pool**')
    st.markdown('**Data pool berisi informasi mengenai pool transporter beserta latitude & longitude nya. Untuk rules pool masih menggunakan hard code.**')
    df_pool
    st.subheader('**Data Speed Region**')
    st.markdown('**Data mapping speed berisi mengenai informasi mapping speed untuk setiap region & vehicle**')
    df_speed_region
    
    df_vehicle_idle, list_jenis_available, list_jenis_order, df_order_routing, df_order_notrouting, df_list_vehicle = get_order_routing_notrouting(df_vehicle_idle, df_order, df_transporter)
    schedule_hard = pd.DataFrame()
    schedule_soft = pd.DataFrame()
    list_order_plan = []
    list_nopol_plan = []
    schedule_vrp_hard = pd.DataFrame()

    st.markdown('**Akan dilakukan Chaining Recommendation menggunakan Hard Constraint**')
    # Akan dilakukan routing sesuai jenis kendaraan
    for jenis_vehicle in list_jenis_order:
        # Assignment akan dibagi untuk setiap tanggal penjemputan-jenis vehicle
        df_order_filtered, dict_assignment, list_assignment_vehicle, temp_vehicle_idle = get_assignment(df_order_routing, df_vehicle_idle, jenis_vehicle)
        # Looping untuk setiap jenis vehicle
        for jenis_assignment in list_assignment_vehicle:
            temp_vehicle_idle = temp_vehicle_idle[~temp_vehicle_idle['nopol'].isin(list_nopol_plan)]
            df_order_filtered = df_order_filtered[~df_order_filtered['id_order'].isin(list_order_plan)]
            if len(temp_vehicle_idle) == 0: # Kalau tidak ada vehicle yang available di-skip
                continue
            df_order_filtered = df_order_filtered[df_order_filtered['jenis_jadwal_enc'] == jenis_assignment].reset_index(drop =True) # Filter sesuai dengan unique id assignment 
            # Create master node. Master node berisi all lokasi beserta variable yang dibutuhkan untuk menjalankan VRP
            df_master_routing, dummy_depot, df_vehicle_idle_depot, df_depot_routing, df_mapping_task = get_master_node(df_order_filtered, df_leadtime_kendaraan, temp_vehicle_idle)
            # Prepare variable untuk routing 
            df_distance_matrix, df_duration_matrix, df_time_window, df_del_drop, jadwal_berangkat, df_vehicle_combination = prepare_variable(df_master_routing, dummy_depot, df_vehicle_idle_depot, df_list_vehicle)

            # Masukkan variable ke data
            distance_matrix = df_distance_matrix.to_numpy()
            duration_matrix = df_duration_matrix.to_numpy()
            timewindow_matrix = df_time_window.set_index('unique_task').T.to_dict('list')
            deldrop_matrix = df_del_drop.T.to_dict('list')
            starts_vehicle = df_vehicle_idle_depot['unique_task'].tolist()
            ends_vehicle = df_vehicle_idle_depot['dummy_depot'].tolist()
            data = {}
            data['distance_matrix'] = distance_matrix
            data['time_matrix'] = duration_matrix 
            data['num_vehicles'] = len(df_vehicle_idle_depot)
            data['starts'] = starts_vehicle
            data['ends'] = ends_vehicle
            data['pickups_deliveries'] = deldrop_matrix
            data['time_windows'] = timewindow_matrix
            data['loading_duration'] = df_master_routing['loading_duration'].tolist()

            # Create manager routing (untuk definisikan jumlah setiap variabel)
            manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                                data['num_vehicles'], data['starts'],
                                                data['ends'])
            # Create Routing Model
            routing = pywrapcp.RoutingModel(manager)
            transit_callback_index = routing.RegisterTransitCallback(distance_callback) # Untuk trace distance kebelakang
            time_callback_index = routing.RegisterTransitCallback(time_callback)  # Untuk trace waktu kebelakang
            routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index) # Yang menjadi objective adalah time

            # Distance dimension
            dimension_name = 'Distance'
            routing.AddDimension(
                transit_callback_index,
                0,  # no slack
                99999999,  # No constraint, tidak diset karena sudah ada time window
                True,
                dimension_name)
            distance_dimension = routing.GetDimensionOrDie(dimension_name)

            # Add Time Windows constraint.
            time = 'Time'
            routing.AddDimension(
                time_callback_index,
                120,  # bisa datang 120 menit sebelum jadwal yg ditentukan
                99999999,  # maximum time per vehicle
                False, 
                time)
            time_dimension = routing.GetDimensionOrDie(time)
            
            # Time window in each location
            for location_idx in data['time_windows'].keys():
                time_window = data['time_windows'][location_idx]
                index = manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

            # Add Penalty
            for index, row in df_del_drop.iterrows():
                routing.AddDisjunction([manager.NodeToIndex(i) for i in (row['unique_task'], row['from'])], 20000, 2)

            # Define delivery-drop matrix
            for location_idx in data['pickups_deliveries']:
                request = data['pickups_deliveries'][location_idx]
                pickup_index = manager.NodeToIndex(request[1])
                delivery_index = manager.NodeToIndex(request[0])
                routing.AddPickupAndDelivery(pickup_index, delivery_index)
                routing.solver().Add(
                    routing.VehicleVar(pickup_index) == routing.VehicleVar(
                        delivery_index))
                routing.solver().Add(
                    time_dimension.CumulVar(pickup_index) <=
                    time_dimension.CumulVar(delivery_index))
                # Force to deliver before next route
                routing.solver().Add(routing.NextVar(pickup_index) == delivery_index)

            # Add vehicle restriction
            unique_node = df_vehicle_combination['unique_task'].unique()
            for node in unique_node:
                temp_restriction = df_vehicle_combination[df_vehicle_combination['unique_task'] == node]['unique_vehicle'].tolist()
                temp_restriction.extend([-1])
                temp_restriction.sort()
                index = manager.NodeToIndex(node)
                routing.VehicleVar(index).SetValues(temp_restriction) 

            # Setting Solver
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
            search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
            search_parameters.time_limit.FromSeconds(1)
            # Prepare Dataframe
            schedule_vrp = df_master_routing.copy().drop(columns={'id_order', 'kota_asal', 'kota_tujuan'})
            list_column_dummy = ['unique_task', 'location','kota' ,'latitude', 'longitude', 'jadwal',
            'jenis_kendaraan', 'from','marker','loading_duration' ,'speed', 'estimasi_selesai', 'constraint_tahun',
            'end_timewindow','start_timewindow', 'vehicle', 'route_order', 'point_distance', 'point_timer','service_time' ,'end_time_min', 'end_time_max', 'route_distance', 'route_time']
            temp_schedule_vrp = pd.DataFrame()
            temp_schedule_vrp = temp_schedule_vrp.reindex(columns =list_column_dummy) 
            schedule_vrp_final = pd.DataFrame()
            solution = routing.SolveWithParameters(search_parameters)
            if solution:
                schedule_vrp_final = print_solution(data, manager, routing, solution,schedule_vrp, temp_schedule_vrp)
            else :
                continue

            if len(schedule_vrp_final) == 0:
                continue
            else :
                schedule_vrp_final,  list_order_plan, list_nopol_plan = rules_pool(schedule_vrp_final, df_vehicle_idle_depot, df_mapping_task, jadwal_berangkat, list_order_plan, list_nopol_plan)
            schedule_vrp_hard = pd.concat([schedule_vrp_hard, schedule_vrp_final], axis = 0).drop_duplicates().reset_index(drop = True)

    # Jika tidak bisa solve dengan hard constraint maka akan dicoba dengan soft constraint
    if len(schedule_vrp_hard) > 0:
        df_routing_detail_report, df_routing_report, df_trip_report, df_vehicle_report, df_not_routing_report, df_vehicle_sisa_report = generate_report(schedule_vrp_hard, df_order, df_vehicle_idle, list_order_plan, list_nopol_plan)
        schedule_hard = df_trip_report.copy()
        temp_schedule_hard = df_routing_detail_report.copy()
        # df_routing_detail_report_copy = df_routing_detail_report.fillna('NaN')
        # df_routing_detail_report_copy = df_routing_detail_report_copy.applymap(str)
        # sheet_routing_hard_constraint.clear()
        # sheet_routing_hard_constraint.update([df_routing_detail_report_copy.columns.values.tolist()] + df_routing_detail_report_copy.fillna("NaN").values.tolist())
        
        df_trip_report_copy = df_trip_report.fillna('NaN')
        df_trip_report_copy = df_trip_report_copy.applymap(str)
        sheet_recommendation_vehicle_hard_constraint.clear()
        sheet_recommendation_vehicle_hard_constraint.update([df_trip_report_copy.columns.values.tolist()] + df_trip_report_copy.fillna("NaN").values.tolist())
    
        # df_not_routing_copy = df_not_routing_report.fillna('NaN')
        # df_not_routing_copy = df_not_routing_copy.applymap(str)
        # sheet_not_feasible_hard_constraint.clear()
        # sheet_not_feasible_hard_constraint.update([df_not_routing_copy.columns.values.tolist()] + df_not_routing_copy.fillna("NaN").values.tolist())
        
        # df_vehicle_sisa_copy = df_vehicle_sisa_report.fillna('NaN')
        # df_vehicle_sisa_copy = df_vehicle_sisa_copy.applymap(str)
        # sheet_remaining_vehicle_hard_constraint.clear()
        # sheet_remaining_vehicle_hard_constraint.update([df_vehicle_sisa_copy.columns.values.tolist()] + df_vehicle_sisa_copy.fillna("NaN").values.tolist())
        
        # st.subheader('**Chaining Routing Recommendation (Hard-Constraint)**')
        # st.markdown('**Berikut adalah rekomendasi rute yang di-sarankan untuk diambil oleh masing-masing kendaraan (menggunakan hard-constraint)**')
        # df_routing_detail_report
        # st.subheader('**Recommendation Vehicle (Hard-Constraint)**')
        # st.markdown('**Berikut adalah rekomendasi kendaraan untuk masing-masing order (menggunakan hard-constraint)**')
        # df_trip_report
        # st.subheader('**Order Not Feasible (Hard-Constraint)**')
        # st.markdown('**Berikut adalah order yang tidak feasible untuk dilakukan Chaining Routing**')
        # df_not_routing_report
        # st.subheader('**Remaining Vehicle (Hard-Constraint)**')
        # st.markdown('**Berikut adalah sisa vehicle yang tidak digunakan**')
        # df_vehicle_sisa_report
        st.markdown('**Hard Constraint Done**')
        
        if len(df_not_routing_report) > 0:
            st.markdown('**Selanjutnya akan dilakukan Chaining Recommendation menggunakan Soft Constraint**')
            # Buat dataframe yang gaada constraint nya
            schedule_vrp_soft = pd.DataFrame()
            # Filter dulu orderan yang tidak bisa multi-trip
            df_vehicle_soft = df_vehicle_idle[df_vehicle_idle['nopol'].isin(df_vehicle_sisa_report['nopol'].unique().tolist())].reset_index(drop = True)
            df_order_soft = df_order[df_order['id_order'].isin(df_not_routing_report['id_order'].unique().tolist())]     
            # Get order an yang jenis kendaraannya tersedia & yang tidak tersedia     
            df_vehicle_soft, list_jenis_available_soft, list_jenis_order_soft, df_order_routing_soft, df_order_notrouting_soft, df_list_vehicle_soft = get_order_routing_notrouting(df_vehicle_soft, df_order_soft, df_transporter)      
            for jenis_vehicle in list_jenis_order_soft:
                df_order_filtered_soft, dict_assignment, list_assignment_vehicle, temp_vehicle_idle_soft = get_assignment(df_order_routing_soft, df_vehicle_soft, jenis_vehicle)
                for jenis_assignment in list_assignment_vehicle:
                    temp_vehicle_idle_soft = temp_vehicle_idle_soft[~temp_vehicle_idle_soft['nopol'].isin(list_nopol_plan)] # Filter vehicle yang belum di-plan
                    df_order_filtered_soft = df_order_filtered_soft[~df_order_filtered_soft['id_order'].isin(list_order_plan)]
                    if len(temp_vehicle_idle_soft) == 0: # Kalau tidak ada vehicle yang available di-skip
                            continue
                    df_order_filtered_soft = df_order_filtered_soft[df_order_filtered_soft['jenis_jadwal_enc'] == jenis_assignment].reset_index(drop =True)  # Filter sesuai dengan unique id assignment          
                    # Create master node. Master node berisi all lokasi beserta variable yang dibutuhkan untuk menjalankan VRP
                    df_master_routing, dummy_depot, df_vehicle_idle_depot, df_depot_routing, df_mapping_task = get_master_node(df_order_filtered_soft, df_leadtime_kendaraan, temp_vehicle_idle_soft)
                    # Prepare variable untuk routing 
                    df_distance_matrix, df_duration_matrix, df_time_window, df_del_drop, jadwal_berangkat, df_vehicle_combination = prepare_variable(df_master_routing, dummy_depot, df_vehicle_idle_depot, df_list_vehicle_soft)

                    # Masukkan variable ke data
                    distance_matrix = df_distance_matrix.to_numpy()
                    duration_matrix = df_duration_matrix.to_numpy()
                    timewindow_matrix = df_time_window.set_index('unique_task').T.to_dict('list')
                    deldrop_matrix = df_del_drop.T.to_dict('list')
                    starts_vehicle = df_vehicle_idle_depot['unique_task'].tolist()
                    ends_vehicle = df_vehicle_idle_depot['dummy_depot'].tolist()
                    data = {}
                    data['distance_matrix'] = distance_matrix
                    data['time_matrix'] = duration_matrix 
                    data['num_vehicles'] = len(df_vehicle_idle_depot)
                    data['starts'] = starts_vehicle
                    data['ends'] = ends_vehicle
                    data['pickups_deliveries'] = deldrop_matrix
                    data['time_windows'] = timewindow_matrix
                    data['loading_duration'] = df_master_routing['loading_duration'].tolist()

                    # Create manager routing (untuk definisikan jumlah setiap variabel)
                    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                                            data['num_vehicles'], data['starts'],
                                                            data['ends'])
                    # Create Routing Model
                    routing = pywrapcp.RoutingModel(manager)
                    transit_callback_index = routing.RegisterTransitCallback(distance_callback) # Untuk trace distance kebelakang
                    time_callback_index = routing.RegisterTransitCallback(time_callback) # Untuk trace waktu kebelakang
                    routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index) # Yang menjadi objective adalah time

                    # Distance dimension
                    dimension_name = 'Distance'
                    routing.AddDimension(
                        transit_callback_index,
                        0,  # no slack
                        99999999,  # No constraint, tidak diset karena sudah ada time window
                        True,  
                        dimension_name)
                    distance_dimension = routing.GetDimensionOrDie(dimension_name)

                    # Add Time Windows constraint.
                    time = 'Time'
                    routing.AddDimension(
                        time_callback_index,
                        99999999, # Slack bebas
                        99999999,  # maximum time per vehicle
                        False, 
                        time)
                    time_dimension = routing.GetDimensionOrDie(time)
                    
                    # Time window in each location
                    for location_idx in data['time_windows'].keys():
                        time_window = data['time_windows'][location_idx]
                        index = manager.NodeToIndex(location_idx)
                        time_dimension.SetCumulVarSoftLowerBound(index, time_window[0],1)
                        time_dimension.SetCumulVarSoftUpperBound(index, time_window[1],1)

                    # Add Penalty
                    for index, row in df_del_drop.iterrows():
                        routing.AddDisjunction([manager.NodeToIndex(i) for i in (row['unique_task'], row['from'])], 20000, 2)

                    # Define delivery-drop matrix    
                    for location_idx in data['pickups_deliveries']:
                        request = data['pickups_deliveries'][location_idx]
                        pickup_index = manager.NodeToIndex(request[1])
                        delivery_index = manager.NodeToIndex(request[0])
                        routing.AddPickupAndDelivery(pickup_index, delivery_index)
                        routing.solver().Add(
                            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                                delivery_index))
                        routing.solver().Add(
                            time_dimension.CumulVar(pickup_index) <=
                            time_dimension.CumulVar(delivery_index))
                        # Force to deliver before next route
                        routing.solver().Add(routing.NextVar(pickup_index) == delivery_index)
                    # Add vehicle restriction
                    unique_node = df_vehicle_combination['unique_task'].unique()
                    for node in unique_node:
                        temp_restriction = df_vehicle_combination[df_vehicle_combination['unique_task'] == node]['unique_vehicle'].tolist()
                        temp_restriction.extend([-1])
                        temp_restriction.sort()
                        index = manager.NodeToIndex(node)
                        routing.VehicleVar(index).SetValues(temp_restriction) 
            
                    # Setting Solver
                    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                    search_parameters.first_solution_strategy = (
                            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
                    search_parameters.local_search_metaheuristic = (
                    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
                    search_parameters.time_limit.FromSeconds(1)

                    # Prepare Dataframe
                    schedule_vrp = df_master_routing.copy().drop(columns={'id_order', 'kota_asal', 'kota_tujuan'})
                    list_column_dummy = ['unique_task', 'location','kota' ,'latitude', 'longitude', 'jadwal',
                'jenis_kendaraan', 'from','marker','loading_duration' ,'speed', 'estimasi_selesai', 'constraint_tahun', 'group_customer',
                'end_timewindow','start_timewindow', 'vehicle', 'route_order', 'point_distance', 'point_timer','service_time' ,'end_time_min', 'end_time_max', 'route_distance', 'route_time']
                    temp_schedule_vrp= pd.DataFrame()
                    schedule_vrp_final = pd.DataFrame()
                    temp_schedule_vrp = temp_schedule_vrp.reindex(columns =list_column_dummy) 
                    solution = routing.SolveWithParameters(search_parameters)
                    if solution:
                        schedule_vrp_final = print_solution(data, manager, routing, solution,schedule_vrp, temp_schedule_vrp)
                    else :
                        continue
                    if len(schedule_vrp_final) == 0:
                        continue
                    else :
                        schedule_vrp_final,  list_order_plan, list_nopol_plan = rules_pool(schedule_vrp_final, df_vehicle_idle_depot, df_mapping_task, jadwal_berangkat, list_order_plan, list_nopol_plan)
                    schedule_vrp_soft = pd.concat([schedule_vrp_soft, schedule_vrp_final], axis = 0).drop_duplicates().reset_index(drop = True)    
            if len(schedule_vrp_soft) > 0:
                # Krn soft constraint bs berangkat kapan saja, maka perlu adjust jadwal berangkat
                for nopol in schedule_vrp_soft['nopol'].unique().tolist():
                    perjalanan = schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 2), 'point_timer'].values[0]
                    index = schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 2)].index
                    schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 1), 'end_time_min_conv'] = (schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 2), 'end_time_min_conv'] - timedelta(minutes = perjalanan))[index.values[0]]
                    schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 1), 'end_time_max_conv'] = (schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 2), 'end_time_max_conv'] - timedelta(minutes = perjalanan))[index.values[0]]
                df_routing_detail_soft_report, df_routing_soft_report, df_trip_soft_report, df_vehicle_soft_report, df_not_routing_soft_report, df_vehicle_sisa_soft_report = generate_report(schedule_vrp_soft, df_order, df_vehicle_idle, list_order_plan, list_nopol_plan)
                schedule_soft = df_trip_soft_report.copy()
                # df_routing_detail_report2_copy = df_routing_detail_soft_report.fillna('NaN')
                # df_routing_detail_report2_copy = df_routing_detail_report2_copy.applymap(str)
                # sheet_routing_soft_constraint.clear()
                # sheet_routing_soft_constraint.update([df_routing_detail_report2_copy.columns.values.tolist()] + df_routing_detail_report2_copy.fillna("NaN").values.tolist())
                
                df_additional_hard = df_trip_soft_report.groupby("no_polisi").filter(lambda x: len(x) == 1)
                df_trip_soft_report = df_trip_soft_report.groupby("no_polisi").filter(lambda x: len(x) > 1)
                # Update hard constraint pakai soft yang len nya 1
                df_trip_report = pd.concat([df_trip_report, df_additional_hard], axis = 0).reset_index(drop = True)
                schedule_hard = df_trip_report.copy()
                df_trip_report_copy = df_trip_report.fillna('NaN')
                df_trip_report_copy = df_trip_report_copy.applymap(str)
                sheet_recommendation_vehicle_hard_constraint.clear()
                sheet_recommendation_vehicle_hard_constraint.update([df_trip_report_copy.columns.values.tolist()] + df_trip_report_copy.fillna("NaN").values.tolist())
                
                df_trip_report2_copy = df_trip_soft_report.fillna('NaN')
                df_trip_report2_copy = df_trip_report2_copy.applymap(str)
                sheet_recommendation_vehicle_soft_constraint.clear()
                sheet_recommendation_vehicle_soft_constraint.update([df_trip_report2_copy.columns.values.tolist()] + df_trip_report2_copy.fillna("NaN").values.tolist())
                
                # df_not_routing2_copy = df_not_routing_soft_report.fillna('NaN')
                # df_not_routing2_copy = df_not_routing2_copy.applymap(str)
                # sheet_not_feasible_soft_constraint.clear()
                # sheet_not_feasible_soft_constraint.update([df_not_routing2_copy.columns.values.tolist()] + df_not_routing2_copy.fillna("NaN").values.tolist())
                
                # df_vehicle_sisa2_copy = df_vehicle_sisa_soft_report.fillna('NaN')
                # df_vehicle_sisa2_copy = df_vehicle_sisa2_copy.applymap(str)
                # sheet_remaining_vehicle_soft_constraint.clear()
                # sheet_remaining_vehicle_soft_constraint.update([df_vehicle_sisa2_copy.columns.values.tolist()] + df_vehicle_sisa2_copy.fillna("NaN").values.tolist())
                # st.subheader('**Chaining Routing Recommendation (Soft-Constraint)**')
                # st.markdown('**Berikut adalah rekomendasi rute yang di-sarankan untuk diambil oleh masing-masing kendaraan (menggunakan Soft-constraint)**')
                # df_routing_detail_soft_report
                
                # st.subheader('**Recommendation Vehicle (Soft-Constraint)**')
                # st.markdown('**Berikut adalah rekomendasi kendaraan untuk masing-masing order (menggunakan Soft-constraint)**')
                # df_trip_soft_report

                # st.subheader('**Order Not Feasible (Soft-Constraint)**')
                # st.markdown('**Berikut adalah order yang tidak feasible untuk dilakukan Chaining Routing (baik dengan soft-constraint maupun hard constraint)**')
                # df_not_routing_soft_report
                
                # st.subheader('**Remaining Vehicle (Soft-Constraint)**')
                # st.markdown('**Berikut adalah sisa vehicle yang tidak digunakan**')
                # df_vehicle_sisa_soft_report
                st.markdown('**Soft Constraint Done**')
            else :
                st.markdown('**Tidak Feasible menggunakan soft constraint**')

        elif len(schedule_hard) == 0 :
            st.markdown('**Karena tidak feasible menggunakan hard-constraint. Selanjutnya akan dilakukan Chaining Recommendation menggunakan Soft Constraint**')  
            # Jika pakai hard constraint tidak bisa maka akan di solve menggunakan soft constraint
            schedule_vrp_soft = pd.DataFrame()
            df_vehicle_soft, list_jenis_available_soft, list_jenis_order_soft, df_order_routing_soft, df_order_notrouting_soft, df_list_vehicle_soft = get_order_routing_notrouting(df_vehicle_idle, df_order, df_transporter)
            # Definisikan list untuk menyimpan order & nopol yang sudah assign
            list_order_plan = []
            list_nopol_plan = []
            # Akan dilakukan routing sesuai jenis kendaraan
            for jenis_vehicle in list_jenis_order_soft:
                # Assignment akan dibagi untuk setiap tanggal penjemputan-jenis vehicle
                df_order_filtered_soft, dict_assignment, list_assignment_vehicle, temp_vehicle_idle_soft = get_assignment(df_order_routing_soft, df_vehicle_soft, jenis_vehicle)
                # Looping untuk setiap jenis vehicle
                for jenis_assignment in list_assignment_vehicle:
                    temp_vehicle_idle_soft = temp_vehicle_idle_soft[~temp_vehicle_idle_soft['nopol'].isin(list_nopol_plan)] # Filter vehicle yang belum di-plan
                    df_order_filtered_soft = df_order_filtered_soft[~df_order_filtered_soft['id_order'].isin(list_order_plan)]
                    if len(temp_vehicle_idle_soft) == 0: # Kalau tidak ada vehicle yang available di-skip
                            continue
                    df_order_filtered_soft = df_order_filtered_soft[df_order_filtered_soft['jenis_jadwal_enc'] == jenis_assignment].reset_index(drop =True)  # Filter sesuai dengan unique id assignment  
                    # Create master node. Master node berisi all lokasi beserta variable yang dibutuhkan untuk menjalankan VRP
                    df_master_routing, dummy_depot, df_vehicle_idle_depot, df_depot_routing, df_mapping_task = get_master_node(df_order_filtered_soft, df_leadtime_kendaraan, temp_vehicle_idle_soft)
                    # Prepare variable untuk routing
                    df_distance_matrix, df_duration_matrix, df_time_window, df_del_drop, jadwal_berangkat, df_vehicle_combination = prepare_variable(df_master_routing, dummy_depot, df_vehicle_idle_depot, df_list_vehicle_soft)
                    
                    # Routing with soft constraint
                    distance_matrix = df_distance_matrix.to_numpy()
                    duration_matrix = df_duration_matrix.to_numpy()
                    timewindow_matrix = df_time_window.set_index('unique_task').T.to_dict('list')
                    deldrop_matrix = df_del_drop.T.to_dict('list')
                    starts_vehicle = df_vehicle_idle_depot['unique_task'].tolist()
                    ends_vehicle = df_vehicle_idle_depot['dummy_depot'].tolist()
                    data = {}
                    data['distance_matrix'] = distance_matrix
                    data['time_matrix'] = duration_matrix 
                    data['num_vehicles'] = len(df_vehicle_idle_depot)
                    data['starts'] = starts_vehicle
                    data['ends'] = ends_vehicle
                    data['pickups_deliveries'] = deldrop_matrix
                    data['time_windows'] = timewindow_matrix
                    data['loading_duration'] = df_master_routing['loading_duration'].tolist()

                    # Create manager routing
                    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                                        data['num_vehicles'], data['starts'],
                                                        data['ends'])
                    # Create Routing Model
                    routing = pywrapcp.RoutingModel(manager)
                    transit_callback_index = routing.RegisterTransitCallback(distance_callback) # Untuk trace distance kebelakang
                    time_callback_index = routing.RegisterTransitCallback(time_callback) # Untuk trace waktu kebelakang
                    routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index) # Yang menjadi objective adalah time

                    # Distance dimension
                    dimension_name = 'Distance'
                    routing.AddDimension(
                        transit_callback_index,
                        0,  # no slack
                        99999999,  # No constraint, tidak diset karena sudah ada time window
                        True, 
                        dimension_name)
                    distance_dimension = routing.GetDimensionOrDie(dimension_name)

                    # Add Time Windows constraint
                    time = 'Time'
                    routing.AddDimension(
                        time_callback_index,
                        99999999,
                        99999999,  # maximum time per vehicle
                        False, 
                        time)
                    time_dimension = routing.GetDimensionOrDie(time)
                    
                    # Time window in each location
                    for location_idx in data['time_windows'].keys():
                        time_window = data['time_windows'][location_idx]
                        index = manager.NodeToIndex(location_idx)
                        time_dimension.SetCumulVarSoftLowerBound(index, time_window[0],1)
                        time_dimension.SetCumulVarSoftUpperBound(index, time_window[1],1)
                    
                    # Add Penalty
                    for index, row in df_del_drop.iterrows():
                        routing.AddDisjunction([manager.NodeToIndex(i) for i in (row['unique_task'], row['from'])], 20000, 2)

                    # Define delivery-drop matrix
                    for location_idx in data['pickups_deliveries']:
                        request = data['pickups_deliveries'][location_idx]
                        pickup_index = manager.NodeToIndex(request[1])
                        delivery_index = manager.NodeToIndex(request[0])
                        routing.AddPickupAndDelivery(pickup_index, delivery_index)
                        routing.solver().Add(
                            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                                delivery_index))
                        routing.solver().Add(
                            time_dimension.CumulVar(pickup_index) <=
                            time_dimension.CumulVar(delivery_index))
                        # Force to deliver before next route
                        routing.solver().Add(routing.NextVar(pickup_index) == delivery_index)

                    # Add vehicle restriction
                    unique_node = df_vehicle_combination['unique_task'].unique()
                    for node in unique_node:
                        temp_restriction = df_vehicle_combination[df_vehicle_combination['unique_task'] == node]['unique_vehicle'].tolist()
                        temp_restriction.extend([-1])
                        temp_restriction.sort()
                        index = manager.NodeToIndex(node)
                        routing.VehicleVar(index).SetValues(temp_restriction) 

                    # Setting Solver
                    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                    search_parameters.first_solution_strategy = (
                            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
                    search_parameters.local_search_metaheuristic = (
                    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
                    search_parameters.time_limit.FromSeconds(1)

                    # Prepare Dataframe
                    schedule_vrp = df_master_routing.copy().drop(columns={'id_order', 'kota_asal', 'kota_tujuan'})
                    list_column_dummy = ['unique_task', 'location','kota' ,'latitude', 'longitude', 'jadwal',
                'jenis_kendaraan', 'from','marker','loading_duration' ,'speed', 'estimasi_selesai', 'constraint_tahun', 'group_customer',
                'end_timewindow','start_timewindow', 'vehicle', 'route_order', 'point_distance', 'point_timer','service_time' ,'end_time_min', 'end_time_max', 'route_distance', 'route_time']
                    temp_schedule_vrp= pd.DataFrame()
                    temp_schedule_vrp = temp_schedule_vrp.reindex(columns =list_column_dummy) 
                    schedule_vrp_final = pd.DataFrame()
                    solution = routing.SolveWithParameters(search_parameters)
                    if solution:
                        schedule_vrp_final = print_solution(data, manager, routing, solution,schedule_vrp, temp_schedule_vrp)
                    else :
                        continue
            
                    if len(schedule_vrp_final) == 0:
                        continue
                    else :
                        schedule_vrp_final,  list_order_plan, list_nopol_plan = rules_pool(schedule_vrp_final, df_vehicle_idle_depot, df_mapping_task, jadwal_berangkat, list_order_plan, list_nopol_plan)
                    schedule_vrp_soft = pd.concat([schedule_vrp_soft, schedule_vrp_final], axis = 0).drop_duplicates().reset_index(drop = True)
            schedule_soft = schedule_vrp_soft.copy() 
            if len(schedule_vrp_soft) > 0: # Jika lebih dari 0 artinya ada order yang bs di multi-trip
                for nopol in schedule_vrp_soft['nopol'].unique().tolist(): # Update jadwal berangkat karena kalau soft constraint jadwal berangkat bebas
                    perjalanan = schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 2), 'point_timer'].values[0]
                    index = schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 2)].index
                    schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 1), 'end_time_min_conv'] = (schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 2), 'end_time_min_conv'] - timedelta(minutes = perjalanan))[index.values[0]]
                    schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 1), 'end_time_max_conv'] = (schedule_vrp_soft.loc[(schedule_vrp_soft['nopol'] == nopol) & (schedule_vrp_soft['route_order'] == 2), 'end_time_max_conv'] - timedelta(minutes = perjalanan))[index.values[0]]
                df_routing_detail_soft_report, df_routing_soft_report, df_trip_soft_report, df_vehicle_soft_report, df_not_routing_soft_report, df_vehicle_sisa_soft_report = generate_report(schedule_vrp_soft, df_order, df_vehicle_idle, list_order_plan, list_nopol_plan)
                schedule_soft = df_trip_soft_report.copy()
                # df_routing_detail_report_copy = df_routing_detail_soft_report.fillna('NaN')
                # df_routing_detail_report_copy = df_routing_detail_report_copy.applymap(str)
                # sheet_remaining_vehicle_soft_constraint.clear()
                # sheet_remaining_vehicle_soft_constraint.update([df_routing_detail_report_copy.columns.values.tolist()] + df_routing_detail_report_copy.fillna("NaN").values.tolist())
                df_additional_hard = df_trip_soft_report.groupby("no_polisi").filter(lambda x: len(x) == 1)
                df_trip_soft_report = df_trip_soft_report.groupby("no_polisi").filter(lambda x: len(x) > 1)
                # Update hard constraint pakai soft yang len nya 1
                schedule_hard = df_additional_hard.copy()
                df_trip_report_copy = df_additional_hard.fillna('NaN')
                df_trip_report_copy = df_trip_report_copy.applymap(str)
                sheet_recommendation_vehicle_hard_constraint.clear()
                sheet_recommendation_vehicle_hard_constraint.update([df_trip_report_copy.columns.values.tolist()] + df_trip_report_copy.fillna("NaN").values.tolist())
                
                df_trip_report_copy = df_trip_soft_report.fillna('NaN')
                df_trip_report_copy = df_trip_report_copy.applymap(str)
                sheet_recommendation_vehicle_soft_constraint.clear()
                sheet_recommendation_vehicle_soft_constraint.update([df_trip_report_copy.columns.values.tolist()] + df_trip_report_copy.fillna("NaN").values.tolist())
                
                # df_not_routing_copy = df_not_routing_soft_report.fillna('NaN')
                # df_not_routing_copy = df_not_routing_copy.applymap(str)
                # sheet_not_feasible_soft_constraint.clear()
                # sheet_not_feasible_soft_constraint.update([df_not_routing_copy.columns.values.tolist()] + df_not_routing_copy.fillna("NaN").values.tolist())
                
                # df_vehicle_sisa_copy = df_vehicle_sisa_soft_report.fillna('NaN')
                # df_vehicle_sisa_copy = df_vehicle_sisa_copy.applymap(str)
                # sheet_remaining_vehicle_soft_constraint.clear()
                # sheet_remaining_vehicle_soft_constraint.update([df_vehicle_sisa_copy.columns.values.tolist()] + df_vehicle_sisa_copy.fillna("NaN").values.tolist())

                # st.subheader('**Chaining Routing Recommendation (Soft-Constraint)**')
                # st.markdown('**Berikut adalah rekomendasi rute yang di-sarankan untuk diambil oleh masing-masing kendaraan (menggunakan Soft-constraint)**')
                # df_routing_detail_soft_report
               
                # st.subheader('**Recommendation Vehicle (Soft-Constraint)**')
                # st.markdown('**Berikut adalah rekomendasi kendaraan untuk masing-masing order (menggunakan Soft-constraint)**')
                # df_trip_soft_report
               
                # st.subheader('**Order Not Feasible (Soft-Constraint)**')
                # st.markdown('**Berikut adalah order yang tidak feasible untuk dilakukan Chaining Routing (baik dengan soft-constraint maupun hard constraint)**')
                # df_not_routing_soft_report

                # st.subheader('**Remaining Vehicle (Soft-Constraint)**')
                # st.markdown('**Berikut adalah sisa vehicle yang tidak digunakan**')
                # df_vehicle_sisa_soft_report
                st.markdown('**Soft Constraint Done**')
                
    recommendation_ranking = pd.DataFrame()
    if len(schedule_hard) > 0:
        # Read Input yang akan digunakan untuk rekomendasi vehicle
        # sheet_id = '1dHMcwlYez_MAthxEX52QspZ0dcI8pnoqMcIWsuYtBd4'
        sheet_id = '1OE3vzXS3uVd2Y96tS165VbEBN7RVHK9EOFcc6onOGGo'
        tab_order = 'order'
        tab_vehicle = 'idle_vehicle'
        tab_mlocation = 'master_location'
        tab_mvehicle = 'master_vehicle' 

        url_order = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_order}"
        url_vehicle = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_vehicle}"
        url_mlocation = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_mlocation}"
        url_mvehicle = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_mvehicle}"

        df_vehicle = pd.read_csv(url_vehicle)
        df_mlocation = pd.read_csv(url_mlocation)
        df_mvehicle = pd.read_csv(url_mvehicle)

        # Preprocessing df_vehicle
        df_vehicle = df_vehicle.dropna(subset=['nopol', 'vehicle_type', 'nama_transporter', 'last_position', 'kota',
        'latitude_last', 'longitude_last']).reset_index(drop = True)
        df_vehicle['nopol'] = df_vehicle['nopol'].fillna(np.nan).astype(str, errors ='ignore')
        schedule_hard['no_polisi'] = schedule_hard['no_polisi'].fillna(np.nan).astype(str, errors ='ignore')
        df_vehicle = df_vehicle[~(df_vehicle['nopol'].isin(schedule_hard['no_polisi'].unique().tolist()))].reset_index(drop = True)
        
        # Year Constraint
        df_mlocation['constraint_tahun'] = pd.to_numeric(df_mlocation['constraint_tahun'], errors='coerce', downcast='integer').fillna(0).astype(int)
        min_year = pd.to_numeric(df_mvehicle['tahun_kendaraan'], errors='coerce').dropna().min()
        df_mvehicle['tahun_kendaraan'] = pd.to_numeric(df_mvehicle['tahun_kendaraan'], errors='coerce', downcast='integer').fillna(min_year).astype(int)
        df_vehicle = df_vehicle.dropna(subset=['nopol', 'vehicle_type', 'nama_transporter', 'last_position', 'kota',
            'latitude_last', 'longitude_last']).drop_duplicates(subset = ['nopol']).reset_index(drop = True)
        df_vehicle = pd.merge(df_vehicle, df_mvehicle[['nopol','plan_awal','active_flag','tahun_kendaraan']], how='left', on='nopol')
        df_vehicle['tahun_kendaraan'] = df_vehicle['tahun_kendaraan'].fillna(min_year).astype(int)
        # Filter kendaraan yang gapunya plan awal & active
        df_vehicle = df_vehicle[(df_vehicle['plan_awal'].isnull()) & (df_vehicle['active_flag'] == True)].reset_index(drop = True)
        
        # Preprocessing
        df_order = pd.read_csv(url_order)
        df_order = df_order.dropna(subset= ['id_order', 'asal', 'latitude_asal', 'longitude_asal', 'tujuan',
    'latitude_tujuan', 'longitude_tujuan', 'jadwal_penjemputan',
    'tanggal_booking', 'jenis_kendaraan', 'kota_asal', 'kota_tujuan']).reset_index(drop = True)
        df_order['id_order'] = df_order['id_order'].fillna(np.nan).astype(str, errors ='ignore')
        schedule_hard['id_order'] = schedule_hard['id_order'].fillna(np.nan).astype(str, errors ='ignore')
        df_order = df_order[~(df_order['id_order'].isin(schedule_hard['id_order'].unique().tolist()))].reset_index(drop = True)
        if len(df_order) == 0 :
            pass
        else :
            # Group customer constraint
            df_order = pd.merge(df_order, df_mlocation[['lokasi','constraint_tahun']],how= 'left' ,left_on='asal', right_on='lokasi').drop(columns='lokasi').rename(columns= {'constraint_tahun' : 'constraint_asal'}) \
            .merge(df_mlocation[['lokasi','constraint_tahun']],how= 'left' ,left_on='tujuan', right_on='lokasi').drop(columns='lokasi').rename(columns= {'constraint_tahun' : 'constraint_tujuan'})
            df_order['constraint_tahun'] = df_order[['constraint_asal', 'constraint_tujuan']].max(axis=1).fillna(0)
            
            # Masukkan ke fungsi
            df_vehicle_combination = vehicle_restriction(df_order, df_vehicle, df_transporter)
            if len(df_vehicle_combination) == 0:
                recommendation_ranking = pd.DataFrame()
            else :
                df_rekomendasi_vehicle_top3 = recommendation_vehicle(df_vehicle_combination)
                recommendation_ranking = df_rekomendasi_vehicle_top3.copy()
                sheet_recommendation_vehicle.clear()
                sheet_recommendation_vehicle.update([df_rekomendasi_vehicle_top3.columns.values.tolist()] + df_rekomendasi_vehicle_top3.fillna("NaN").values.tolist())
    else :
        # Read Input yang akan digunakan untuk rekomendasi vehicle
        # sheet_id = '1dHMcwlYez_MAthxEX52QspZ0dcI8pnoqMcIWsuYtBd4'
        sheet_id = '1OE3vzXS3uVd2Y96tS165VbEBN7RVHK9EOFcc6onOGGo'
        tab_order = 'order'
        tab_vehicle = 'idle_vehicle'
        tab_mlocation = 'master_location'
        tab_mvehicle = 'master_vehicle' 

        url_order = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_order}"
        url_vehicle = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_vehicle}"
        url_mlocation = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_mlocation}"
        url_mvehicle = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_mvehicle}"

        df_vehicle = pd.read_csv(url_vehicle)
        df_mlocation = pd.read_csv(url_mlocation)
        df_mvehicle = pd.read_csv(url_mvehicle)

        # Preprocessing df_vehicle
        df_vehicle = df_vehicle.dropna(subset=['nopol', 'vehicle_type', 'nama_transporter', 'last_position', 'kota',
        'latitude_last', 'longitude_last']).reset_index(drop = True)
        df_vehicle['nopol'] = df_vehicle['nopol'].fillna(np.nan).astype(str, errors ='ignore')

        # Year Constraint
        df_mlocation['constraint_tahun'] = pd.to_numeric(df_mlocation['constraint_tahun'], errors='coerce', downcast='integer').fillna(0).astype(int)
        min_year = pd.to_numeric(df_mvehicle['tahun_kendaraan'], errors='coerce').dropna().min()
        df_mvehicle['tahun_kendaraan'] = pd.to_numeric(df_mvehicle['tahun_kendaraan'], errors='coerce', downcast='integer').fillna(min_year).astype(int)
        df_vehicle = df_vehicle.dropna(subset=['nopol', 'vehicle_type', 'nama_transporter', 'last_position', 'kota',
            'latitude_last', 'longitude_last']).drop_duplicates(subset = ['nopol']).reset_index(drop = True)
        df_vehicle = pd.merge(df_vehicle, df_mvehicle[['nopol','plan_awal','active_flag','tahun_kendaraan']], how='left', on='nopol')
        df_vehicle['tahun_kendaraan'] = df_vehicle['tahun_kendaraan'].fillna(min_year).astype(int)
        # Filter kendaraan yang gapunya plan awal & active
        df_vehicle = df_vehicle[(df_vehicle['plan_awal'].isnull()) & (df_vehicle['active_flag'] == True)].reset_index(drop = True)
        
        # Preprocessing
        df_order = pd.read_csv(url_order)
        df_order = df_order.dropna(subset= ['id_order', 'asal', 'latitude_asal', 'longitude_asal', 'tujuan',
    'latitude_tujuan', 'longitude_tujuan', 'jadwal_penjemputan',
    'tanggal_booking', 'jenis_kendaraan', 'kota_asal', 'kota_tujuan']).reset_index(drop = True)
        df_order['id_order'] = df_order['id_order'].fillna(np.nan).astype(str, errors ='ignore')
        
        # Group customer constraint
        df_order = pd.merge(df_order, df_mlocation[['lokasi','constraint_tahun']],how= 'left' ,left_on='asal', right_on='lokasi').drop(columns='lokasi').rename(columns= {'constraint_tahun' : 'constraint_asal'}) \
        .merge(df_mlocation[['lokasi','constraint_tahun']],how= 'left' ,left_on='tujuan', right_on='lokasi').drop(columns='lokasi').rename(columns= {'constraint_tahun' : 'constraint_tujuan'})
        df_order['constraint_tahun'] = df_order[['constraint_asal', 'constraint_tujuan']].max(axis=1).fillna(0)

        # Masukkan ke fungsi
        df_vehicle_combination = vehicle_restriction(df_order, df_vehicle, df_transporter)
        if len(df_vehicle_combination) == 0:
            recommendation_ranking = pd.DataFrame()
        else :
            df_rekomendasi_vehicle_top3 = recommendation_vehicle(df_vehicle_combination)
            recommendation_ranking = df_rekomendasi_vehicle_top3.copy()
            sheet_recommendation_vehicle.clear()
            sheet_recommendation_vehicle.update([df_rekomendasi_vehicle_top3.columns.values.tolist()] + df_rekomendasi_vehicle_top3.fillna("NaN").values.tolist())
    
    # Update ke google sheet (sisa order dan sisa kendaraan)
    order_planning = []
    vehicle_planning = []
    if len(schedule_hard) > 0:
        order_planning.extend(schedule_hard['id_order'].unique().tolist())
        vehicle_planning.extend(schedule_hard['no_polisi'].unique().tolist())
    if len(recommendation_ranking) > 0:
        order_planning.extend(recommendation_ranking['id_order'].unique().tolist())
        vehicle_planning.extend(recommendation_ranking['nopol'].unique().tolist())
    not_routing = df_order[~df_order['id_order'].isin(order_planning)].reset_index(drop = True)
    not_routing_copy = not_routing.fillna('NaN')
    not_routing_copy = not_routing_copy.applymap(str)
    sheet_not_routing.clear()
    sheet_not_routing.update([not_routing_copy.columns.values.tolist()] + not_routing_copy.fillna("NaN").values.tolist())

    vehicle_sisa = df_vehicle[~df_vehicle['nopol'].isin(vehicle_planning)].reset_index(drop = True)
    vehicle_sisa_copy = vehicle_sisa.fillna('NaN')
    vehicle_sisa_copy = vehicle_sisa_copy.applymap(str)
    sheet_vehicle_sisa.clear()
    sheet_vehicle_sisa.update([vehicle_sisa_copy.columns.values.tolist()] + vehicle_sisa_copy.fillna("NaN").values.tolist())


    # Print Elapsed Time
    end_hitung = tm.time()
    elapsed_time_second = round((end_hitung - start_hitung),2)
    elapsed_time_minutes = round(elapsed_time_second/60,2)
    len_filtered = len(pd.read_csv(url_order).dropna(subset = ['id_order', 'asal','latitude_asal', 'longitude_asal', 'tujuan', 'latitude_tujuan', 'longitude_tujuan', 'jadwal_penjemputan']).reset_index(drop = True))

    html_len_filtered = f"""
    <style>
    p.a {{
    font: bold 18px Courier;
    }}
    </style>
    <p class="a">Len Dataset : {len_filtered}</p>
    """
    html_elapsed_time_second = f"""
    <style>
    p.a {{
    font: bold 18px Courier;
    }}
    </style>
    <p class="a">Elapsed Time : {elapsed_time_second} Second</p>
    """
    html_elapsed_time_minutes = f"""
    <style>
    p.a {{
    font: bold 18px Courier;
    }}
    </style>
    <p class="a">Elapsed Time : {elapsed_time_minutes} Minutes</p>
    """
    st.markdown('**Finished Running**')
    st.markdown(html_len_filtered, unsafe_allow_html=True)
    st.markdown(html_elapsed_time_second, unsafe_allow_html=True)
    st.markdown(html_elapsed_time_minutes, unsafe_allow_html=True)

else :
    st.markdown('**Engine Not Running**')
