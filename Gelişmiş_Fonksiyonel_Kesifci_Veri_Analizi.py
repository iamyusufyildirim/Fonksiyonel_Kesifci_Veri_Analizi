                                        #############################################
                                        # Gelişmiş Fonksiyonel Keşifci Veri Analizi #
                                        #############################################

# 1. Genel Resim
# 2. Kategorik ve Sayısal Değişken Analizi
# 3. Eksik Değer Analizi



# Gerekli kütüphane importları ve bazı görsel ayarlamalar
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", lambda x : "%.3f" % x)


# Titanic veri setinin projeye dahil edilmesi
df = sns.load_dataset("titanic")


##################
# 1. Genel Resim #
##################

def check_df(dataframe, head=10):
    """
    İlgili veri setine ait satır bilgisi, boyut bilgisi, değişken isimleri gibi
    özet bilgiler sunarak veri setine ilk bakışı gerçekleştirmenize olanak tanır.

    Parameters
    ----------
    dataframe : dataframe
         özetlemesi yapılacak ilgili veri seti.
    head : int
        Veri setine ait ilk kaç gözlem biriminin gösterileceği bilgisi.

    """
    print("###################################")
    print(f"#### İlk {head} Gözlem Birimi ####")
    print("###################################")
    print(dataframe.head(head), "\n\n")

    print("###################################")
    print("###### Veri Seti Boyut Bilgisi ####")
    print("###################################")
    print(dataframe.shape, "\n\n")

    print("###################################")
    print("######## Değişken İsimleri ########")
    print("###################################")
    print(dataframe.columns, "\n\n")

    print("###################################")
    print("####### Eksik Değer Var mı? #######")
    print("###################################")
    print(dataframe.isnull().values.any(), "\n\n")

    print("###################################")
    print("##### Betimsel İstatistikler ######")
    print("###################################")
    print(dataframe.describe().T, "\n\n")

    print("###################################")
    print("### Veri Seti Hakkında Bilgiler ###")
    print("###################################")
    print(dataframe.info())

check_df(dataframe=df)



#####################################################
# 2. Kategorik ve Sayısal Değişkenlerin Yakalanması #
#####################################################

def grab_col_names(dataframe, cat_th=10, car_th=20, num_th=20, plot=False):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin raporlama işlemini gerçekleştirir.

    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen veri seti.
    cat_th : int, float
        kategorik değişkenler için sınıf eşik değeri.
    car_th : int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri.
    num_th : int, float
        numerik değişkenler için sınıf eşik değeri.
    plot : bool

    Returns
    -------
        cat_cols: list
            Kategorik değişkenlerin listesi
        cat_but_car: list
            Kategorik görünümlü kardinal değişken listesi
        num_cols: list
            Numerik değişkenlerin listesi

    Notes
    -----
    plot argümanının True olması durumunda ilgili kategorik ve numerik değişkenler görselleştirilecektir.
    """

    # Veri setindeki kategorik değişkenleri filtreleme işlemini gerçekleştirir.
    # Kod bloğunun Türkçe meali:
    # Veri setindeki değişkenlerde gez ve ilgili değişkenin eşsiz sınıf sayısı cat_th değerinden küçükse ve
    # ilgili değişkenin tip bilgisi "category", "object", "bool" ise bu değişkeni cat_cols'ta sakla.
    cat_cols = [col for col in dataframe.columns
                if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["category", "object", "bool"]]

    # Veri setindeki numerik görünümlü ancak kategorik olan değişkenleri filtreler.
    num_but_cat = [col for col in dataframe.columns
                   if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

    cat_cols += num_but_cat

    # Veri setindeki kardinal değişkenleri filtreleme işlemini gerçekleştirir.
    cardinal = [col for col in dataframe.columns
                if dataframe[col].nunique() > car_th and dataframe[col].dtypes in ["category", "object", "bool"]]

    # plot argümanı True'ysa
    if plot:
        # cat cols içerisinde gez
        for cat in cat_cols:
            # bool veri tipinde görselleştirme kısmında hata alındığı için ilgili hatanın giderilmesi için tip bilgisi değiştirildi.
            if dataframe[cat].dtypes == "bool":
                dataframe[cat] = dataframe[cat].astype(int)
            else:
                print(dataframe[cat].value_counts())
                print("##########################")

                sns.countplot(x=dataframe[cat], data=dataframe)
                plt.show(block=True)

    # Veri setindeki numerik değişkenleri filtreleme işlemini gerçekleştirir.
    num_cols = [col for col in dataframe.columns
                if dataframe[col].nunique() > num_th and dataframe[col].dtypes in ["int", "float"]]

    if plot:
        for num in num_cols:
            print(dataframe[num].describe().T)
            print("##########################")

            dataframe[num].hist()
            plt.show(block=True)


    # PrettyTable nesnesini oluşturma
    tablo = PrettyTable()
    tablo.field_names = ["", "Toplam Değişken Sayısı", "Değişken İsimleri"]

    # Veri eklemek
    tablo.add_row(["Kategorik Değişken", len(cat_cols) , cat_cols])
    tablo.add_row(["Kardinal Değişken", "-", cardinal])
    tablo.add_row(["Sayısal Değişken", len(num_cols), num_cols])
    tablo.add_row(["Diğer Değişken", dataframe.shape[1] - (len(cat_cols) + len(num_cols)), [col for col in dataframe.columns
                                                                                            if col not in cat_cols and num_cols]])
    # Tabloyu yazdırma
    print(tablo)


    return cat_cols, cardinal, num_cols




grab_col_names(dataframe=df)



##########################
# 3. Eksik Değer Analizi #
##########################
# veri setindeki eksik değerleri tablo formatında gösterecek bir fonksiyon tanımlıyoruz.
def missing_values_table(dataframe, na_columns=False):
    """
    İlgili veri setindeki eksik değerlere sahip değişkenleri değer ve oran olacak formatta çıktısını üretir.

    Parameters
    ----------
    dataframe : dataframe
        Eksik değer kontrolü yapılmak istenilen veri seti
    na_columns : bool
        Eksik değere sahip değişken isimlerinin barındırıldığı
        listenin çıktısının istenilip istenilmediği bilgisi.
        bool -> True or False


    Returns
    -------
    na_columns : list
        Eksik gözlem birimine sahip olan değişken isimlerinin listesi.
    """
    # Eksik değere sahip olan değişkenlerin filtrelenmesi
    na_columns = [col for col in dataframe.columns if
                  dataframe[col].isnull().sum() > 0]
    # Eksik değere sahip olan değişkenlere ait değerler
    missing_values = (dataframe[na_columns].isnull().sum()).sort_values(ascending=False)
    # Eksik değere sahip olan değişkenlerin gözlem birim sayısına yüzdelik oranı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # İlgili tablonun oluşturulması
    table = pd.concat([missing_values, ratio], axis=1, keys=["Değer", "%"])
    print(table)

    if na_columns:
        return na_columns

missing_values_table(dataframe=df)

