from fastapi import FastAPI, Body, Path, Query
from pydantic import BaseModel, Field
from enum import Enum
from datetime import date, time, datetime, timedelta
from typing import List, Optional
import logging
import logging.config
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import io
import numpy as np
from scipy.interpolate import griddata
from data_loader import load_data_from_path
from logging_conf import log_config
from visualizations import create_skyplot, create_time_series_plot
from map_creator import create_map, create_contour_map
import base64
# Initialize logging
try:
    logging.config.dictConfig(log_config)
except Exception as e:
    logging.error(f"Failed to configure logging: {e}")

logger = logging.getLogger(__name__)

app = FastAPI(
    title="S4 Data Plotter API",
    description="API for retrieving and visualizing S4 data for the Dynamic S4 Data Plotter application.",
    version="1.0.0",
)

# Enums
class VisualizationType(str, Enum):
    Map = "Map"
    TimeSeries = "TimeSeries"
    Skyplot = "Skyplot"

class MapType(str, Enum):
    ScatterHeatmap = "Scatter/Heatmap"
    TEC = "TEC"

class MapStyle(str, Enum):
    OpenStreetMap = "open-street-map"
    CartoDarkMatter = "carto-darkmatter"
    CartoPositron = "carto-positron"
    StamenTerrain = "stamen-terrain"
    StamenToner = "stamen-toner"
    StamenWatercolor = "stamen-watercolor"

# Models
class DataPreview(BaseModel):
    data: List[dict]

class FilteredDataParams(BaseModel):
    file_name: str
    selected_date: date
    selected_time: time
    window: int = Field(10, ge=1, le=30)
    latitude_range: str
    longitude_range: str
    s4_threshold: float = Field(0.0, ge=0.0, le=1.0)
    svid: Optional[List[int]] = None

class FilteredData(BaseModel):
    data: List[dict]
    svid_options: List[int]

class PlotlyVisualization(BaseModel):
    data: List[dict]
    layout: dict

class MatplotlibVisualization(BaseModel):
    image: str

class Visualization(BaseModel):
    type: str
    data: PlotlyVisualization | MatplotlibVisualization

# Global DataFrame
df = None

def add_time(selected_time, window_minutes):
    logger.info(f"Adding time window of {window_minutes} minutes to {selected_time}")
    
    dt = datetime.combine(date.today(), selected_time)
    dt += timedelta(minutes=window_minutes)
    if dt.date() > date.today():
        dt = datetime.combine(date.today(), time(23, 59, 59))
    return dt.time()

# Function to filter the DataFrame
def filter_dataframe(df: pl.DataFrame, params: FilteredDataParams) -> pl.DataFrame:
    logger.info(f"Filtering DataFrame with parameters: {params}")
    
    svid = params.svid if params.svid else df["SVID"].unique().to_list()
    latitude_range = tuple(map(float, params.latitude_range.split(',')))
    longitude_range = tuple(map(float, params.longitude_range.split(',')))
    s4_threshold = params.s4_threshold
    time_window = (
        datetime.combine(params.selected_date, params.selected_time),
        datetime.combine(params.selected_date, add_time(params.selected_time, params.window))
    )
    
    filter_conditions = [
        pl.col("SVID").is_in(svid),
        pl.col("Latitude").is_between(latitude_range[0], latitude_range[1]),
        pl.col("Longitude").is_between(longitude_range[0], longitude_range[1]),
        pl.col("S4") >= s4_threshold,
    ]
    
    if time_window:
        filter_conditions.extend([pl.col("IST_Time") >= time_window[0], pl.col("IST_Time") <= time_window[1]])
    
    for condition in filter_conditions:
        df = df.filter(condition)
        logger.debug(f"Applied filter condition: {condition}")
    
    logger.info(f"Filtered DataFrame shape: {df.shape}")
    
    return df

@app.get("/files", response_model=List[str])
async def get_files() -> List[str]:
    logger.info("Retrieving list of files")
    
    try:
        # Assuming files are stored in a directory named 'data'
        import os
        files = os.listdir('data')
        logger.info(f"Files found: {files}")
        return files
    except Exception as e:
        logger.error(f"Failed to retrieve list of files: {e}")
        raise

@app.get("/data_preview/{file_name}", response_model=DataPreview)
async def get_data_preview(
    file_name: str = Path(..., description="Name of the file to preview")
) -> DataPreview:
    logger.info(f"Previewing data from file: {file_name}")
    
    try:
        df = load_data_from_path(f"data/{file_name}")
        if df is not None:
            data_dict = df.head(10).to_dict()
            # Convert dict of Series to list of dicts
            data_list = [dict(zip(data_dict.keys(), values)) for values in zip(*data_dict.values())]
            logger.info("Data preview generated successfully")
            
            return DataPreview(data=data_list)
        else:
            logger.error(f"Failed to load data from file: {file_name}")
            raise ValueError("Failed to load data")
    except Exception as e:
        logger.error(f"Failed to generate data preview: {e}")
        raise

@app.get("/filtered_data", response_model=FilteredData)
async def get_filtered_data(
    file_name: str = Query(..., description="Name of the file to analyze"),
    selected_date: date = Query(
        ..., description="Selected date for filtering (YYYY-MM-DD)"
    ),
    selected_time: time = Query(
        ..., description="Selected time for filtering (HH:MM:SS)"
    ),
    window: int = Query(10, ge=1, le=30, description="Time window in minutes"),
    latitude_range: str = Query(
        ..., description="Latitude range (min,max)", regex=r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?$"
    ),
    longitude_range: str = Query(
        ..., description="Longitude range (min,max)", regex=r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?$"
    ),
    s4_threshold: float = Query(0.0, ge=0.0, le=1.0, description="S4 threshold value"),
    svid: Optional[List[int]] = Query(None, description="List of SVID values to include")
) -> FilteredData:
    logger.info(f"Filtering data from file: {file_name} with parameters: {selected_date}, {selected_time}, {window}, {latitude_range}, {longitude_range}, {s4_threshold}, {svid}")
    
    try:
        df = load_data_from_path(f"data/{file_name}")
        if df is not None:
            filtered_df = filter_dataframe(df, FilteredDataParams(
                file_name=file_name,
                selected_date=selected_date,
                selected_time=selected_time,
                window=window,
                latitude_range=latitude_range,
                longitude_range=longitude_range,
                s4_threshold=s4_threshold,
                svid=svid
            ))
            
            # Convert dict of Series to list of dicts
            data_dict = filtered_df.to_dict()
            data_list = [dict(zip(data_dict.keys(), values)) for values in zip(*data_dict.values())]
            
            logger.info("Filtered data generated successfully")
            return FilteredData(data=data_list, svid_options=df["SVID"].unique().to_list())
        else:
            logger.error(f"Failed to load data from file: {file_name}")
            raise ValueError("Failed to load data")
    except Exception as e:
        logger.error(f"Failed to generate filtered data: {e}")
        raise


@app.get("/visualization/{viz_type}", response_model=Visualization)
async def get_visualization(
    viz_type: VisualizationType = Path(..., description="Type of visualization"),
    file_name: str = Query(..., description="Name of the file to visualize"),
    filtered_data_params: FilteredDataParams = Body(
        ..., description="JSON object containing all filtered data parameters"
    ),
    map_type: Optional[str] = Query(
        None, description="Type of map visualization (required for Map viz_type)"
    ),
    map_style: Optional[str] = Query(
        None, description="Style of the map (required for Map viz_type)"
    ),
    marker_size: Optional[int] = Query(
        None, description="Size of markers for Scatter/Heatmap"
    ),
    heatmap_size: Optional[int] = Query(
        None, description="Size of heatmap for Scatter/Heatmap"
    ),
    color_scale: Optional[str] = Query(
        None, description="Color scale for the visualization"
    ),
    bin_heatmap: Optional[bool] = Query(None, description="Whether to bin the heatmap"),
    selected_svid: Optional[int] = Query(
        None, description="Selected SVID for Time Series visualization"
    ),
) -> Visualization:
    logger.info(f"Generating visualization of type: {viz_type} for file: {file_name}")
    
    try:
        df = load_data_from_path(f"data/{file_name}")
        if df is not None:
            filtered_df = filter_dataframe(df, filtered_data_params)
            
            if viz_type == VisualizationType.Map:
                fig = create_map(
                    df=filtered_df,
                    lat="Latitude",
                    lon="Longitude",
                    color="Vertical Scintillation Amplitude",
                    size=None,
                    map_type=map_type,
                    map_style=map_style,
                    zoom=3.6,
                    marker_size=marker_size or 5,
                    heatmap_size=heatmap_size or 20,
                    color_scale=color_scale,
                    bin_heatmap=bin_heatmap or False,
                )
                
                logger.info("Starting to render plot as image")
                buffer = io.BytesIO()
                fig.write_image(buffer, format="png", engine="kaleido")
                buffer.seek(0)
                logger.info("Finished rendering plot as image")
                
                image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                logger.info("Visualization generated successfully")
                return Visualization(type=viz_type.value, data=MatplotlibVisualization(image=image_data))
            
            elif viz_type == VisualizationType.TimeSeries:
                fig = create_time_series_plot(filtered_df, selected_svid)
                logger.info("Time series visualization generated successfully")
                return Visualization(type=viz_type.value, data=PlotlyVisualization(data=fig.data, layout=fig.layout))
            
            elif viz_type == VisualizationType.Skyplot:
                fig = create_skyplot(filtered_df)
                logger.info("Skyplot visualization generated successfully")
                return Visualization(type=viz_type.value, data=PlotlyVisualization(data=fig.data, layout=fig.layout))
        else:
            logger.error(f"Failed to load data from file: {file_name}")
            raise ValueError("Failed to load data")
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)