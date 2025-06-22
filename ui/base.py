from typing import Any, Dict

import streamlit as st

# TODO: Replace with actual unique songs from the Dataset
total_songs = {
    "Blinding Lights — The Weeknd",
    "Castle on the Hill — Ed Sheeran",
    "Levitating — Dua Lipa",
    "Shape of You — Ed Sheeran",
    "Uptown Funk — Mark Ronson ft. Bruno Mars",
    "drivers license — Olivia Rodrigo",
    "Can't Stop the Feeling! — Justin Timberlake",
    "Rolling in the Deep — Adele",
    "Bad Guy — Billie Eilish",
    "Take Me to Church — Hozier"
}


def create_ui(config: Dict[str, Any]):
    """Create the actual StreamLit UI for user interation.

    Args:
        config (Dict[str, Any]): The configuration loaded from the yml file.
    """
    st.title(config["app"]["name"])
    query = st.text_input(
        "Enter the song to search and select to get recommendation:")
    if query:
        # Case-insensitive filter over total_songs
        matches = [song for song in total_songs if query.lower()
                   in song.lower()]
        if matches:
            selected_songs = st.multiselect("Available songs:", matches)
            if selected_songs:
                # TODO: Replace with actual recommendation engine call.
                st.write("You selected:", selected_songs)
        else:
            st.write("No matching songs found.")
