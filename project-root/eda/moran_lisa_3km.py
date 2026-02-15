import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local


# =========================
# 你只需要改这里的配置
# =========================
IN_CSV = "data/crimes_2024.csv"     # 你的犯罪点数据（必须有 latitude/longitude）
CELL_M = 3000                      # 3km 网格
PERMUTATIONS = 999
ALPHA = 0.05

# 裁剪模式：
# 1) "clip"：把网格裁成市界形状（边缘格子会变成不规则多边形，但地图最“像论文”）
# 2) "centroid"：只保留“格子中心点在市界内”的格子（保持完整方格，更规整）
CLIP_MODE = "clip"  # 改成 "centroid" 也行

# 是否叠加社区边界（更丰富，但会多几条线）
OVERLAY_COMMUNITY_AREAS = False


# =========================
# 数据来源（在线 GeoJSON）
# =========================
CITY_BOUNDARY_GEOJSON_URL = "https://data.cityofchicago.org/resource/qqq8-j68g.geojson"
COMMUNITY_AREAS_GEOJSON_URL = "https://data.cityofchicago.org/resource/cauq-8yn6.geojson"


def make_grid(bounds, cell_size_m: int, crs):
    """根据 bounds 生成规则网格（方格 polygon）。"""
    minx, miny, maxx, maxy = bounds
    pad = cell_size_m
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)

    polys = [box(x, y, x + cell_size_m, y + cell_size_m) for x in xs for y in ys]
    return gpd.GeoDataFrame({"geometry": polys}, crs=crs)


def build_grid_counts(in_csv: str, cell_m: int = 3000) -> gpd.GeoDataFrame:
    """把犯罪点聚合到网格，输出 grid(count, geometry)。"""
    df = pd.read_csv(in_csv)
    df = df.dropna(subset=["latitude", "longitude"])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )

    # Chicago 常用 UTM Zone 16N（单位米）
    gdf_m = gdf.to_crs("EPSG:26916")

    grid = make_grid(gdf_m.total_bounds, cell_m, gdf_m.crs)

    # 点落格计数
    joined = gpd.sjoin(gdf_m[["geometry"]], grid, predicate="within", how="left")
    counts = joined.groupby("index_right").size()

    grid["count"] = 0
    grid.loc[counts.index, "count"] = counts.values
    return grid


def load_boundary(url: str) -> gpd.GeoDataFrame:
    """读取边界 GeoJSON，并尽量修复几何。"""
    bnd = gpd.read_file(url)
    # 轻量修复，避免少数自相交导致 clip/overlay 报错
    bnd["geometry"] = bnd.geometry.buffer(0)
    # 保留有效几何
    bnd = bnd[bnd.is_valid & ~bnd.is_empty].copy()
    return bnd


def clip_grid_to_city(grid: gpd.GeoDataFrame, city_bnd: gpd.GeoDataFrame, mode: str = "clip") -> gpd.GeoDataFrame:
    """裁剪网格到市界。"""
    city_bnd = city_bnd.to_crs(grid.crs)

    if mode == "centroid":
        # 保持完整方格：用中心点筛选
        city_geom = city_bnd.geometry.unary_union
        keep = grid.centroid.within(city_geom)
        out = grid.loc[keep].copy()
        return out.reset_index(drop=True)

    # 默认 clip：边缘会被裁成不规则多边形
    try:
        out = gpd.clip(grid, city_bnd)
    except Exception:
        # clip 偶尔会遇到拓扑异常，退化用 intersection
        out = gpd.overlay(grid, city_bnd, how="intersection")

    out = out.reset_index(drop=True)
    return out


def compute_moran_lisa(grid: gpd.GeoDataFrame, value_col="count", permutations=999):
    """计算 Global Moran's I + Local Moran (LISA)。"""
    # 对计数做 log1p，减轻长尾
    y = np.log1p(grid[value_col].astype(float).values)

    # Queen 邻接（共享边或角），网格常用
    w = Queen.from_dataframe(grid, use_index=False)  # 避免 FutureWarning
    w.transform = "R"

    mi = Moran(y, w, permutations=permutations)
    lisa = Moran_Local(y, w, permutations=permutations)
    return mi, lisa


def plot_lisa_cluster(grid: gpd.GeoDataFrame,
                      lisa: Moran_Local,
                      city_bnd: gpd.GeoDataFrame,
                      alpha=0.05,
                      out_png="output_lisa_cluster_cityclip.png",
                      community_bnd: gpd.GeoDataFrame | None = None):
    """画 LISA 聚类图并叠加市界线。"""
    g = grid.copy()
    sig = lisa.p_sim < alpha

    # LISA quadrant:
    # 1 HH, 2 LH, 3 LL, 4 HL
    q = lisa.q

    g["lisa_cat"] = "Not significant"
    g.loc[sig & (q == 1), "lisa_cat"] = "HH (hot spot cluster)"
    g.loc[sig & (q == 3), "lisa_cat"] = "LL (cold spot cluster)"
    g.loc[sig & (q == 4), "lisa_cat"] = "HL (high-low outlier)"
    g.loc[sig & (q == 2), "lisa_cat"] = "LH (low-high outlier)"

    ax = g.plot(column="lisa_cat", categorical=True, legend=True, figsize=(10, 10))

    # 叠加边界线
    city_bnd = city_bnd.to_crs(g.crs)
    city_bnd.boundary.plot(ax=ax, linewidth=1.2)

    if community_bnd is not None:
        community_bnd = community_bnd.to_crs(g.crs)
        community_bnd.boundary.plot(ax=ax, linewidth=0.4)

    ax.set_axis_off()
    plt.title(f"LISA cluster map | cell={CELL_M/1000:.0f}km | alpha={alpha} | clip={CLIP_MODE}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.show()
    print(f"Saved figure: {out_png}")


def main():
    # 1) 网格计数
    grid = build_grid_counts(IN_CSV, cell_m=CELL_M)

    # 2) 读市界并裁剪
    city_bnd = load_boundary(CITY_BOUNDARY_GEOJSON_URL)
    grid_city = clip_grid_to_city(grid, city_bnd, mode=CLIP_MODE)

    # 如果裁剪后为空（极少见：文件路径错/投影错），直接报错
    if len(grid_city) == 0:
        raise RuntimeError("Clipped grid is empty. Check input CSV, CRS, or boundary loading.")

    # 3) 计算 Moran / LISA（在裁剪后的网格上做）
    mi, lisa = compute_moran_lisa(grid_city, value_col="count", permutations=PERMUTATIONS)

    print("=== Global Moran's I (on log1p(count)) ===")
    print(f"I = {mi.I:.4f}")
    print(f"p(sim) = {mi.p_sim:.4g}  (permutations={mi.permutations})")
    print(f"z(sim) = {mi.z_sim:.4f}")

    # 4) 可选：读社区边界叠加
    community_bnd = None
    if OVERLAY_COMMUNITY_AREAS:
        community_bnd = load_boundary(COMMUNITY_AREAS_GEOJSON_URL)

    # 5) 画图
    plot_lisa_cluster(
        grid_city,
        lisa,
        city_bnd=city_bnd,
        alpha=ALPHA,
        out_png="output_lisa_cluster_cityclip.png",
        community_bnd=community_bnd
    )


if __name__ == "__main__":
    main()
