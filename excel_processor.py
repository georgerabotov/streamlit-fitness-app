import streamlit as st
import pandas as pd
import hashlib
import time
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

def generate_safe_id(text, prefix="data"):
    """Generate a safe, ASCII-compliant ID for Pinecone vectors"""
    # Create a hash of the text
    hash_object = hashlib.md5(text.encode('utf-8'))
    # Get the hexadecimal representation (guaranteed to be ASCII)
    hex_dig = hash_object.hexdigest()
    # Return a prefixed ID to make it more identifiable
    return f"{prefix}-{hex_dig[:16]}"

def excel_tab_content(pc, pineconeRegion, client):
    """Function to render the Excel upload tab content"""
    st.header("Upload Excel Data")
    st.write("Upload an Excel file with fitness program data to add to your knowledge base.")
    
    # File upload for Excel
    excel_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"], key="excel_uploader")
    
    if excel_file:
        st.write(f"File '{excel_file.name}' uploaded successfully!")
        
        # Process button
        if st.button("Process Excel", key="process_excel"):
            try:
                # Read Excel file
                df = pd.read_excel(excel_file)
                
                # Check if required columns exist
                required_columns = [
                    "GroupId",
                    "Goal",
                    "DaysPerWeek",
                    "KnowledgeLevel",
                    "SplitType",
                    "WorkoutDay",
                    "WorkoutStructure",
                    "WorkoutExplanation",
                    "FemaleWorkoutStructure",
                    "FemaleAdditionalComments",
                    "MediaUrl"
                ]

                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                else:
                    # Create program-matrix index if it doesn't exist
                    if 'program-matrix-dec' not in pc.list_indexes().names():
                        pc.create_index(
                            name='program-matrix-dec',
                            dimension=1536,  # Dimension for text-embedding-ada-002
                            metric='cosine',
                            spec=ServerlessSpec(
                                cloud='aws',
                                region=pineconeRegion
                            )
                        )
                    
                    # Connect to the program-matrix index
                    program_index = pc.Index('program-matrix-dec')
                    
                    # Process each row
                    total_rows = len(df)
                    progress_bar = st.progress(0.0)
                    
                    for i, row in df.iterrows():
                        # Extract values from each column
                        groupId = str(row['GroupId'])
                        goal = str(row['Goal'])
                        days_per_week = str(row['DaysPerWeek'])
                        knowledge_level = str(row['KnowledgeLevel'])
                        split_type = str(row['SplitType'])
                        workout_day = str(row['WorkoutDay'])
                        workout_structure = str(row['WorkoutStructure'])
                        female_workout_structure = str(row['FemaleWorkoutStructure'])
                        female_additional_comments = str(row['FemaleAdditionalComments'])
                        workout_explanation = str(row['WorkoutExplanation'])
                        media_url = str(row['MediaUrl'])

                        # Create a structured text from the row data for vectorization
                        structured_text = (
                            f"GroupId: {groupId}. "
                            f"Goal: {goal}. "
                            f"Days Per Week: {days_per_week}. "
                            f"Knowledge Level: {knowledge_level}. "  
                            f"Split Type: {split_type}. "  
                            f"Workout Day: {workout_day}. "
                            f"Workout Structure: {workout_structure}."
                            f"Female Workout Structure: {female_workout_structure}."
                            f"Female Additional Comments: {female_additional_comments}."
                            f"Workout Explanation: {workout_explanation}"
                            f"Media Url: {media_url}"
                        )

                        
                        # Generate embedding for the structured text
                        response = client.embeddings.create(
                            model="text-embedding-ada-002",
                            input=structured_text
                        )
                        embedding = response.data[0].embedding
                        
                        # Generate a safe ID
                        safe_id = generate_safe_id(structured_text, prefix=f"program-{i}")
                        
                        # Store in Pinecone with each column as separate metadata
                        program_index.upsert([
                            {
                                "id": safe_id,
                                "values": embedding,
                                "metadata": {
                                    "group_id": groupId,
                                    "goal": goal,
                                    "days_per_week": days_per_week,
                                    "knowledge_level": knowledge_level,
                                    "split_type": split_type,
                                    "workout_day": workout_day,
                                    "workout_structure": workout_structure,
                                    "female_workout_structure": female_workout_structure,
                                    "female_additional_comments": female_additional_comments,
                                    "workout_explanation": workout_explanation,
                                    "media_url": media_url,
                                    "full_text": structured_text  # Also store the full text for reference
                                }
                            }
                        ])

                        
                        # Update progress
                        progress = (i + 1) / total_rows
                        progress_bar.progress(progress)
                        
                        # Small delay to prevent rate limiting
                        time.sleep(0.1)
                        
                    st.success(f"Successfully processed {total_rows} rows from Excel and stored in program-matrix-dev index!")
                    
                    # Show a sample of how the data is structured in Pinecone
                    st.subheader("Data Structure in Pinecone")
                    st.write("Each row is stored with the following structure:")
                    st.code("""
{
    "id": "program-123abc...",
    "values": [0.123, 0.456, ...],  # 1536-dimensional vector
    "metadata": {
        "goal": "Weight Loss",
        "days_per_week": "3",
        "target_area": "Full Body",
        "knowledge_level": "Beginner",
        "outcome": "Lose 10 pounds in 8 weeks",
        "full_text": "Goal: Weight Loss. Days Per Week: 3. Target Area: Full Body. Knowledge Level: Beginner. Outcome: Lose 10 pounds in 8 weeks."
    }
}
                    """)
                    
            except Exception as e:
                st.error(f"Error processing Excel file: {str(e)}")
                st.exception(e)  # Show detailed error information
