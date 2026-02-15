import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

def make_grid(bounds, cell_size_m: int, crs):
    minx, miny, maxx, maxy = bounds
    pad = cell_size_m
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)

    polys = [box(x, y, x + cell_size_m, y + cell_size_m) for x in xs for y in ys]
    return gpd.GeoDataFrame({"geometry": polys}, crs=crs)

def main(in_csv="data/crimes_2024.csv", cell_m=3000):
    df = pd.read_csv(in_csv)
    df = df.dropna(subset=["latitude", "longitude"])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )

    # project to m（Chicago UTM 16N）
    gdf_m = gdf.to_crs("EPSG:26916")

    grid = make_grid(gdf_m.total_bounds, cell_m, gdf_m.crs)

    # space linking：point fall in to which grids
    joined = gpd.sjoin(gdf_m[["geometry"]], grid, predicate="within", how="left")
    counts = joined.groupby("index_right").size()

    grid["count"] = 0
    grid.loc[counts.index, "count"] = counts.values

    # for more clear, using log1p increasing gap/difference
    grid["log_count"] = np.log1p(grid["count"])

    ax = grid.plot(column="log_count", figsize=(10, 10), legend=True)
    ax.set_axis_off()
    plt.title(f"Chicago crimes (3km grid) log1p(count)")

    plt.show()

if __name__ == "__main__":
    main()
