import asyncio
from datetime import datetime, timedelta

import dotenv
import httpx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import httpx
import streamlit as st
from custom_css import custom_css_str

BACKEND_URL = dotenv.get_key(dotenv.find_dotenv(), "BACKEND_URL")


async def fetch_data(client, endpoint, params=None):
    response = await client.get(f"{BACKEND_URL}/{endpoint}", params=params)
    response.raise_for_status()
    return response.json()


async def main():
    st.set_page_config(layout="wide", page_title="Dynamic S4 Data Plotter")

    st.markdown(
        custom_css_str,
        unsafe_allow_html=True,
    )
    st.title("Dynamic S4 Data Plotter")

    async with httpx.AsyncClient() as client:
        # Get available files
        available_files = await fetch_data(client, "files")

        current_file_selection = st.sidebar.selectbox(
            "Select file", options=available_files
        )

        if current_file_selection:
            viz_container = st.empty()
            data_container = st.empty()
            download_container = st.empty()

            # Get data preview
            data_preview = await fetch_data(
                client, f"data_preview/{current_file_selection}"
            )

            with data_container.container():
                st.write("Data Preview:")
                st.write(data_preview)

            st.sidebar.header("Filtering Options and Visualization Options")

            # Sidebar inputs
            selected_date = st.sidebar.date_input("Select Date")
            selected_time = st.sidebar.time_input("Select Time")
            window = st.sidebar.slider(
                "Time Window", value=10, max_value=30, min_value=1
            )
            latitude_range = st.sidebar.slider("Latitude Range", 0, 90, (0, 90))
            longitude_range = st.sidebar.slider(
                "Longitude Range", -180, 180, (-180, 180)
            )
            s4_threshold = st.sidebar.slider("S4 Threshold", 0.0, 1.0, 0.0)

            viz_type = st.sidebar.selectbox(
                "Select Visualization", ["Map", "Time Series", "Skyplot"]
            )

            # Get filtered data
            filtered_data_params = {
                "file_name": current_file_selection,
                "selected_date": selected_date.isoformat(),
                "selected_time": selected_time.isoformat(),
                "window": window,
                "latitude_range": f"{latitude_range[0]},{latitude_range[1]}",
                "longitude_range": f"{longitude_range[0]},{longitude_range[1]}",
                "s4_threshold": s4_threshold,
            }
            filtered_data = await fetch_data(
                client, "filtered_data", filtered_data_params
            )

            # Visualization options
            viz_params = {
                "viz_type": viz_type,
                "file_name": current_file_selection,
                "filtered_data_params": filtered_data_params,
            }

            if viz_type == "Map":
                map_type = st.sidebar.selectbox(
                    "Select Map Type", ["Scatter/Heatmap", "TEC"], index=0
                )
                map_style = st.sidebar.selectbox(
                    "Select Map Style",
                    [
                        "open-street-map",
                        "carto-darkmatter",
                        "carto-positron",
                        "stamen-terrain",
                        "stamen-toner",
                        "stamen-watercolor",
                    ],
                )
                viz_params.update(
                    {
                        "map_type": map_type,
                        "map_style": map_style,
                    }
                )

                if map_type == "Scatter/Heatmap":
                    marker_size = st.sidebar.slider("Marker Size", 1, 20, 10)
                    heatmap_size = st.sidebar.slider("Heatmap Size", 1, 80, 40)
                    bin_heatmap = st.sidebar.toggle("Bin heatmap")
                    viz_params.update(
                        {
                            "marker_size": marker_size,
                            "heatmap_size": heatmap_size,
                            "bin_heatmap": bin_heatmap,
                        }
                    )

                color_scale = st.sidebar.selectbox(
                    "Color Scale", ["Viridis", "Plasma", "Inferno", "Magma"]
                )
                viz_params["color_scale"] = color_scale

            elif viz_type == "Time Series":
                selected_svid = st.selectbox(
                    "Select SVID", options=filtered_data["svid_options"]
                )
                viz_params["selected_svid"] = selected_svid

            # Get visualization
            viz_data = await fetch_data(client, f"visualization/{viz_type}", viz_params)

            # Display visualization
            with viz_container.container():
                if viz_type == "Map":
                    if map_type == "TEC":
                        st.pyplot(viz_data["figure"])
                    else:
                        st.plotly_chart(
                            go.Figure(viz_data["figure"]), use_container_width=True
                        )
                elif viz_type in ["Time Series", "Skyplot"]:
                    st.plotly_chart(
                        go.Figure(viz_data["figure"]), use_container_width=True
                    )

            # Download button
            with download_container.container():
                st.download_button(
                    label="Download Visualization as PNG",
                    data=viz_data["png_data"],
                    file_name=f"s4_visualization_{viz_type}_{map_type if viz_type=='Map' else ''}.png",
                    mime="image/png",
                )

    st.markdown(
        """
    ## Help
    - **Map**: Shows S4 data on a map. Use 'Scatter' for individual points or 'Heatmap' for density.
    - **Time Series**: Displays S4 values over time for a selected satellite.
    - **Skyplot**: Shows satellite positions with color-coded S4 values.
    - Use the filters to narrow down the data shown in the visualizations.
    """
    )


if __name__ == "__main__":
    asyncio.run(main())
