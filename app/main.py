import streamlit as st
import os
import glob
from typing import List, Dict, Any
from services.chatbot import Chatbot

# Configure page
st.set_page_config(
    page_title="Local AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize chatbot with shared connection and enhanced caching
@st.cache_resource
def init_chatbot():
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chat.db")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(project_root, "models", "llama-2-7b.gguf")
    model_path = os.getenv("MODEL_PATH", default_model_path)
    
    chatbot = Chatbot(db_path)
    
    # Load model if path exists
    if os.path.exists(model_path):
        if chatbot.load_model(model_path):
            st.success(f"Model loaded: {model_path}")
        else:
            st.error(f"Failed to load model: {model_path}")
    else:
        st.warning(f"Model not found at: {model_path}")
    
    return chatbot

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_available_models() -> List[str]:
    """Get list of available GGUF models"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    if os.path.exists(models_dir):
        models = glob.glob(os.path.join(models_dir, "*.gguf"))
        return [os.path.basename(model) for model in models]
    return []

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_knowledge_stats_cached(chatbot_id: str) -> Dict[str, Any]:
    """Cached version of knowledge stats to improve performance"""
    try:
        stats = chatbot.get_knowledge_stats()
        # Convert to dictionary for backward compatibility with UI code
        return stats.to_dict()
    except Exception as e:
        print(f"Error getting knowledge stats: {e}")
        return {"document_count": 0, "rag_enabled": False, "vector_extension": False}

@st.cache_data(ttl=30)  # Cache for 30 seconds  
def get_file_stats_cached(chatbot_id: str) -> Dict[str, Any]:
    """Cached version of file stats to improve performance"""
    try:
        stats = chatbot.get_file_stats()
        # Ensure no None values
        return {
            "total_files": stats.get("total_files", 0) or 0,
            "total_size": stats.get("total_size", 0) or 0,
            "file_types": stats.get("file_types", {}) or {}
        }
    except Exception as e:
        print(f"Error getting file stats: {e}")
        return {"total_files": 0, "total_size": 0, "file_types": {}}

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_vector_stats_cached(chatbot_id: str) -> Dict[str, Any]:
    """Cached version of vector stats to improve performance"""
    try:
        return chatbot.vector.get_vector_stats()
    except Exception as e:
        print(f"Error getting vector stats: {e}")
        return {"vector_enabled": False, "vector_initialized": False, "documents_with_embeddings": 0}

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_chat_stats_cached(chatbot_id: str) -> Dict[str, Any]:
    """Cached version of chat stats to improve performance"""
    try:
        return chatbot.get_chat_stats()
    except Exception as e:
        print(f"Error getting chat stats: {e}")
        return {"total_sessions": 0, "total_messages": 0, "last_activity": None}

# Initialize session state
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Chat"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "show_chat_loader" not in st.session_state:
    st.session_state.show_chat_loader = False
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "last_stats_update" not in st.session_state:
    st.session_state.last_stats_update = 0
if "chatbot_id" not in st.session_state:
    # Generate a unique ID for this session to use with caching
    import time
    st.session_state.chatbot_id = str(int(time.time()))
if "is_loading_chat" not in st.session_state:
    st.session_state.is_loading_chat = False
if "is_generating_response" not in st.session_state:
    st.session_state.is_generating_response = False

# Initialize chatbot
chatbot = init_chatbot()

# Main title
st.title("🤖 Local AI Chatbot")

# Tab navigation
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Knowledge", "⚙️ System"])

# Tab 1: Chat Interface
with tab1:
    st.header("💬 Chat Interface")
    
    # Model status and settings
    col1, col2 = st.columns([6, 1])
    
    with col1:
        # Status indicators
        if chatbot.is_ready():
            # Get current model name from session state or model info
            current_model = st.session_state.get("selected_model", "Unknown Model")
            if current_model and current_model != "Unknown Model":
                st.write(f"Current model: 🤖 {current_model}")
            else:
                # Fallback to getting model info from chatbot
                try:
                    model_info = chatbot.get_model_info().to_dict()
                    model_path = model_info.get("model_path", "")
                    if model_path:
                        model_name = os.path.basename(model_path)
                        st.write(f"Current model: 🤖 {model_name}")
                    else:
                        st.write("Current model: 🤖 Model Ready")
                except:
                    st.write("Current model: 🤖 Model Ready")
            
            # Knowledge base status on separate line
            knowledge_stats = chatbot.get_knowledge_stats().to_dict()
            doc_count = knowledge_stats['document_count']
            if doc_count > 0:
                st.write(f"Current knowledge base: 📚 {doc_count} documents")
            else:
                st.write("Current knowledge base: 📚 No documents")
        else:
            st.write("Current model: ⚠️ No Model Loaded")
            st.write("Current knowledge base: 📚 No documents")
    
    with col2:
        # Settings button with popover
        if st.button("⚙️", help="Settings", key="settings_button"):
            st.session_state.show_settings = not st.session_state.get("show_settings", False)
    
    # Settings popover content
    if st.session_state.get("show_settings", False):
        with st.expander("⚙️ Settings", expanded=True):
            settings_col1, settings_col2 = st.columns(2)
            
            with settings_col1:
                st.subheader("Model Selection")
                available_models = get_available_models()
                if available_models:
                    selected_model = st.selectbox(
                        "Select Model",
                        available_models,
                        index=0 if st.session_state.selected_model is None else available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
                        help="Choose from available GGUF models",
                        key="model_selector_settings"
                    )
                    
                    if selected_model != st.session_state.selected_model:
                        st.session_state.selected_model = selected_model
                        # Load new model
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        model_path = os.path.join(project_root, "models", selected_model)
                        if os.path.exists(model_path):
                            with st.spinner("Loading model..."):
                                if chatbot.load_model(model_path):
                                    st.success(f"Model loaded: {selected_model}")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to load model: {selected_model}")
                else:
                    st.warning("No GGUF models found in models/ directory")
            
            with settings_col2:
                st.subheader("Generation Settings")
                # Sampling mode
                if chatbot.is_ready():
                    sampling_mode = st.selectbox(
                        "Sampling Mode",
                        ["Greedy", "Temperature", "Top-P"],
                        help="Text generation strategy",
                        key="sampling_mode_settings"
                    )
                    
                    # Advanced sampling parameters
                    if sampling_mode != "Greedy":
                        temperature = st.slider(
                            "Temperature", 
                            0.1, 2.0, 0.8, 0.1,
                            help="Controls randomness",
                            key="temp_settings"
                        )
                        
                        if sampling_mode == "Top-P":
                            top_p = st.slider(
                                "Top P", 
                                0.1, 0.95, 0.9, 0.05,
                                help="Nucleus sampling",
                                key="top_p_settings"
                            )
            
            # Close settings button
            if st.button("Close Settings", type="secondary"):
                st.session_state.show_settings = False
                st.rerun()
    
    # Chat management buttons - use container to prevent layout shifts
    st.divider()
    with st.container():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🗑️ Clear Chat", help="Clear current conversation", key="clear_chat_btn", disabled=st.session_state.is_loading_chat or st.session_state.is_generating_response):
                st.session_state.chat_messages = []
                if chatbot.is_ready():
                    chatbot.reset_conversation()
                st.rerun()

        with col2:
            if st.button("💾 Save Chat", help="Save current conversation", key="save_chat_btn", disabled=st.session_state.is_loading_chat or st.session_state.is_generating_response):
                if st.session_state.chat_messages:
                    # Generate a smart title from first user message
                    first_user_msg = next((msg for msg in st.session_state.chat_messages if msg["role"] == "user"), None)
                    if first_user_msg:
                        title = first_user_msg["content"][:50] + ("..." if len(first_user_msg["content"]) > 50 else "")
                    else:
                        from datetime import datetime
                        title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"

                    try:
                        chat_uuid = chatbot.save_chat(st.session_state.chat_messages, title)
                        if chat_uuid:
                            st.success(f"Chat saved: {chat_uuid[:8]}...")
                        else:
                            st.error("Failed to save chat")
                    except Exception as e:
                        st.error(f"Error saving chat: {e}")
                else:
                    st.info("No messages to save")

        with col3:
            if st.button("🔄 Reset State", help="Reset conversation state", key="reset_state_btn", disabled=st.session_state.is_loading_chat or st.session_state.is_generating_response):
                if chatbot.is_ready():
                    chatbot.reset_conversation()
                    st.success("Conversation state reset!")

        with col4:
            # Chat loading with modal dialog
            if st.button("📋 Load Chat", help="Load previous conversation", key="load_chat_btn", disabled=st.session_state.is_loading_chat or st.session_state.is_generating_response):
                st.session_state.show_chat_loader = True
    
    # Chat loader modal (shown when button is clicked)
    if st.session_state.get("show_chat_loader", False):
        with st.expander("📋 Load Previous Chat", expanded=True):
            # Use a container to prevent layout shifts
            with st.container():
                chat_sessions = chatbot.get_chat_sessions(20)

                if chat_sessions:
                    st.write("**Recent Conversations:**")

                    for session in chat_sessions:
                        col1, col2, col3 = st.columns([3, 1, 1])

                        with col1:
                            # Show session info
                            st.write(f"**{session['title']}**")
                            st.caption(f"{session['message_count']} messages • {session['updated_at']}")

                        with col2:
                            if st.button("Load", key=f"load_{session['id']}", type="primary", disabled=st.session_state.is_loading_chat):
                                # Set loading state to prevent multiple clicks
                                st.session_state.is_loading_chat = True

                                # Show loading indicator
                                with st.spinner("Loading chat..."):
                                    try:
                                        loaded_messages = chatbot.load_chat(session['id'])
                                        if loaded_messages:
                                            # Update state atomically
                                            st.session_state.chat_messages = loaded_messages
                                            st.session_state.show_chat_loader = False
                                            st.session_state.is_loading_chat = False
                                            st.success(f"Loaded chat: {session['title'][:30]}...")
                                            st.rerun()
                                        else:
                                            st.session_state.is_loading_chat = False
                                            st.error("Failed to load chat messages")
                                    except Exception as e:
                                        st.session_state.is_loading_chat = False
                                        st.error(f"Error loading chat: {e}")

                        with col3:
                            if st.button("Delete", key=f"del_{session['id']}", type="secondary", disabled=st.session_state.is_loading_chat):
                                with st.spinner("Deleting chat..."):
                                    if chatbot.delete_chat_session(session['id']):
                                        st.success("Chat deleted!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete chat")
                
                    # Close button
                    if st.button("Close", type="secondary", disabled=st.session_state.is_loading_chat):
                        st.session_state.show_chat_loader = False
                        st.rerun()
                else:
                    st.info("No saved conversations found.")
                    if st.button("Close", disabled=st.session_state.is_loading_chat):
                        st.session_state.show_chat_loader = False
                        st.rerun()
    
    # Chat interface
    st.divider()
    
    # Create a dedicated chat area that's isolated from dynamic content above
    # Use a fixed-height container to prevent layout shifts
    chat_container = st.container()
    with chat_container:
        # Show loading indicator if chat is being loaded
        if st.session_state.is_loading_chat:
            st.info("🔄 Loading chat...")
        elif st.session_state.is_generating_response:
            st.info("🤖 Generating response...")

        # Display chat history in a stable container
        if st.session_state.chat_messages:
            # Render messages without keys to prevent duplicate components
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            # Show placeholder when no messages
            st.info("👋 Start a conversation! Ask me anything about your uploaded documents.")
    
    # Ensure chat input is always at the bottom with proper spacing
    st.markdown("---")  # Visual separator

    # Disable input during loading or generation
    input_disabled = st.session_state.is_loading_chat or st.session_state.is_generating_response

    if prompt := st.chat_input("What's the capital of Italy?", key="chat_input", disabled=input_disabled):
        if not chatbot.is_ready():
            st.error("Please load a model first!")
        else:
            # Set generation state to prevent multiple submissions
            st.session_state.is_generating_response = True

            # Add user message to history immediately
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            # Generate assistant response without intermediate rerun
            try:
                # Configure sampling based on settings
                sampling_mode = st.session_state.get("sampling_mode_settings", "Greedy")
                if sampling_mode == "Temperature":
                    temp = st.session_state.get("temp_settings", 0.8)
                    chatbot.configure_sampling(temperature=temp)
                elif sampling_mode == "Top-P":
                    temp = st.session_state.get("temp_settings", 0.8)
                    top_p = st.session_state.get("top_p_settings", 0.9)
                    chatbot.configure_sampling(temperature=temp, top_p=top_p)
                else:  # Greedy
                    chatbot.configure_sampling(temperature=0.0)

                # Generate response using streaming
                full_response = ""
                for token in chatbot.send_message_stream(prompt):
                    full_response += token

                # Add assistant message to history
                st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

                # Clear generation state and rerun to display both messages
                st.session_state.is_generating_response = False
                st.rerun()

            except Exception as e:
                st.error(f"Error generating response: {e}")
                fallback_response = "I'm having trouble responding right now. Please try again."
                st.session_state.chat_messages.append({"role": "assistant", "content": fallback_response})
                st.session_state.is_generating_response = False
                st.rerun()
    


# Tab 2: Knowledge Management
with tab2:
    st.header("📚 Knowledge Management")
    
    # Knowledge base statistics
    col1, col2, col3, col4 = st.columns(4)
    
    # Use cached stats for better performance
    knowledge_stats = get_knowledge_stats_cached(st.session_state.chatbot_id)
    file_stats = get_file_stats_cached(st.session_state.chatbot_id)
    
    with col1:
        st.metric("Documents", knowledge_stats['document_count'])
    
    with col2:
        st.metric("Text Chunks", knowledge_stats.get('chunk_count', 0))
    
    with col3:
        total_size = file_stats.get('total_size', 0) or 0
        st.metric("Total Size", f"{total_size:,} bytes" if total_size > 0 else "0 bytes")
    
    with col4:
        vector_status = "✅" if knowledge_stats['vector_extension'] else "❌"
        st.metric("Vector Extension", vector_status)
    
    st.divider()
    
    # File upload section
    st.subheader("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['txt', 'md', 'py', 'js', 'ts', 'html', 'csv', 'json'],
        help="Supported formats: Text, Markdown, Code files, CSV, JSON"
    )
    
    if uploaded_files:
        upload_col1, upload_col2 = st.columns([3, 1])
        
        with upload_col1:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size:,} bytes)")
        
        with upload_col2:
            if st.button("🚀 Process All Files", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                successful_uploads = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {uploaded_file.name}...")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        file_content = uploaded_file.read()
                        file_type = uploaded_file.type or "text/plain"
                        
                        file_id = chatbot.upload_file(file_content, uploaded_file.name, file_type)
                        if file_id:
                            successful_uploads += 1
                        
                        # Reset file pointer for potential re-processing
                        uploaded_file.seek(0)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                if successful_uploads > 0:
                    st.success(f"Successfully processed {successful_uploads} out of {len(uploaded_files)} files!")
                    st.rerun()
                else:
                    st.error("No files were successfully processed.")
    
    st.divider()
    
    # Manual text input
    st.subheader("✏️ Add Text Knowledge")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        knowledge_text = st.text_area(
            "Enter text to add to knowledge base",
            placeholder="Paste or type text content here...",
            height=150
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("➕ Add Knowledge", type="primary", disabled=not knowledge_text.strip()):
            if knowledge_text.strip():
                try:
                    chunk_ids = chatbot.add_knowledge(knowledge_text.strip())
                    st.success(f"Added {len(chunk_ids)} chunks to knowledge base!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add knowledge: {e}")
    
    st.divider()
    
    # View uploaded files
    st.subheader("📄 Uploaded Files")
    
    uploaded_files_list = chatbot.get_uploaded_files(20)
    
    if uploaded_files_list:
        # File type filter
        all_file_types = list(set(f['file_type'] for f in uploaded_files_list))
        selected_file_types = st.multiselect(
            "Filter by file type",
            all_file_types,
            default=all_file_types,
            help="Select file types to display"
        )
        
        # Filter files
        filtered_files = [f for f in uploaded_files_list if f['file_type'] in selected_file_types]
        
        if filtered_files:
            for file_info in filtered_files:
                with st.expander(f"📄 {file_info['filename']} ({file_info['file_type']})"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**Size:** {file_info['file_size']:,} bytes")
                        st.write(f"**Uploaded:** {file_info['upload_date']}")
                    
                    with col2:
                        st.write(f"**Type:** {file_info['file_type']}")
                        st.write(f"**ID:** {file_info['id']}")
                    
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{file_info['id']}", type="secondary"):
                            if chatbot.delete_uploaded_file(file_info['id']):
                                st.success("File deleted!")
                                st.rerun()
                            else:
                                st.error("Failed to delete file")
        else:
            st.info("No files match the selected filters.")
    else:
        st.info("No files uploaded yet. Use the upload section above to add documents.")
    
    # Knowledge base management
    st.divider()
    st.subheader("🔧 Knowledge Base Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clear All Knowledge", type="secondary"):
            if chatbot.clear_knowledge():
                st.success("Knowledge base cleared!")
                st.rerun()
            else:
                st.error("Failed to clear knowledge base")
    
    with col2:
        if knowledge_stats['vector_extension'] and knowledge_stats['documents_with_embeddings'] > 0:
            if st.button("⚡ Optimize Vectors", type="secondary"):
                try:
                    chatbot.vector.quantize_vectors()
                    chatbot.vector.preload_quantized_vectors()
                    st.success("Vector optimization complete!")
                except Exception as e:
                    st.error(f"Vector optimization failed: {e}")

# Tab 3: System Information
with tab3:
    st.header("⚙️ System Information")
    
    # Model Information
    if chatbot.is_ready():
        st.subheader("🤖 Model Information")
        
        model_info = chatbot.get_model_info().to_dict()

        col1, col2 = st.columns(2)

        with col1:
            st.json({
                "Status": model_info.get("status", "Unknown"),
                "Model Path": model_info.get("model_path", "Unknown"),
                "Parameters": f"{model_info.get('n_params', 'Unknown'):,}" if isinstance(model_info.get('n_params'), (int, float)) else model_info.get('n_params', 'Unknown'),
                "Context Length": f"{model_info.get('n_ctx_train', 'Unknown'):,}" if isinstance(model_info.get('n_ctx_train'), (int, float)) else model_info.get('n_ctx_train', 'Unknown'),
                "Embedding Dimension": model_info.get('n_embd', 'Unknown'),
            })

        with col2:
            st.json({
                "Layers": model_info.get('n_layer', 'Unknown'),
                "Heads": model_info.get('n_head', 'Unknown'),
                "Model Size": f"{model_info.get('size', 'Unknown'):,} bytes" if isinstance(model_info.get('size'), (int, float)) else model_info.get('size', 'Unknown'),
                "Description": model_info.get('description', 'Unknown'),
                "Chat Active": model_info.get('chat_active', False)
            })
        
        st.divider()
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        # Get current statistics (using cached versions for performance)
        knowledge_stats = get_knowledge_stats_cached(st.session_state.chatbot_id)
        file_stats = get_file_stats_cached(st.session_state.chatbot_id)
        vector_stats = get_vector_stats_cached(st.session_state.chatbot_id)
        chat_stats = get_chat_stats_cached(st.session_state.chatbot_id)
        
        with perf_col1:
            st.metric(
                "Knowledge Documents",
                knowledge_stats['document_count'],
                help="Total documents in knowledge base"
            )
        
        with perf_col2:
            st.metric(
                "Chat Sessions", 
                chat_stats['total_sessions'],
                help="Total saved chat conversations"
            )
        
        with perf_col3:
            st.metric(
                "Chat Messages", 
                chat_stats['total_messages'],
                help="Total messages across all chats"
            )
        
        with perf_col4:
            quantized_status = "✅" if vector_stats.get('quantized', False) else "❌"
            st.metric(
                "Vectors Quantized", 
                quantized_status,
                help="Whether vectors are optimized for fast search"
            )
        
        st.divider()
        
        # System capabilities
        st.subheader("🔧 System Capabilities")
        
        cap_col1, cap_col2 = st.columns(2)
        
        with cap_col1:
            st.write("**AI Capabilities:**")
            try:
                ai_version = chatbot.get_version()
                st.write(f"- AI Extension: ✅ v{ai_version}")
            except:
                st.write("- AI Extension: ❌ Not available")
            
            st.write(f"- Text Generation: {'✅' if chatbot.is_ready() else '❌'}")
            st.write(f"- Embedding Generation: {'✅' if chatbot.is_ready() else '❌'}")
            st.write(f"- RAG Support: {'✅' if knowledge_stats['rag_enabled'] else '⚪'}")
        
        with cap_col2:
            st.write("**Vector Capabilities:**")
            st.write(f"- Vector Extension: {'✅' if vector_stats['vector_enabled'] else '❌'}")
            st.write(f"- Vector Database: {'✅' if vector_stats.get('vector_initialized', False) else '❌'}")
            st.write(f"- Similarity Search: {'✅' if vector_stats['vector_enabled'] else '❌'}")
            st.write(f"- Vector Quantization: {'✅' if vector_stats.get('quantized', False) else '❌'}")
        
        st.divider()
        
        # Advanced settings
        st.subheader("⚙️ Advanced Settings")
        
        with st.expander("🎛️ Sampling Configuration"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                temperature = st.slider(
                    "Temperature", 
                    0.0, 2.0, 0.8, 0.1,
                    help="Controls randomness in text generation"
                )
            
            with col2:
                top_p = st.slider(
                    "Top P", 
                    0.1, 0.95, 0.9, 0.05,
                    help="Nucleus sampling parameter"
                )
            
            with col3:
                top_k = st.slider(
                    "Top K", 
                    1, 100, 40, 1,
                    help="Top-K sampling parameter"
                )
            
            if st.button("Apply Sampling Settings"):
                try:
                    chatbot.configure_sampling(temperature=temperature, top_p=top_p, top_k=top_k)
                    st.success("Sampling settings updated!")
                except Exception as e:
                    st.error(f"Failed to update settings: {e}")
        
        with st.expander("🗄️ Database Information"):
            import os
            db_path = chatbot.db_manager.db_path
            if os.path.exists(db_path):
                db_size = os.path.getsize(db_path)
                st.write(f"**Database Path:** `{db_path}`")
                st.write(f"**Database Size:** {db_size:,} bytes")
                
                # Show table information
                try:
                    with chatbot.db_manager.get_connection() as conn:
                        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                        st.write("**Tables:**")
                        for table in tables:
                            table_name = table[0]
                            count_result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                            count = count_result[0] if count_result else 0
                            st.write(f"- {table_name}: {count:,} rows")
                except Exception as e:
                    st.error(f"Failed to get database info: {e}")
            else:
                st.error(f"Database not found at: {db_path}")
        
        with st.expander("🔄 System Actions"):
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                if st.button("🔄 Reset All State", type="secondary"):
                    try:
                        chatbot.reset_conversation()
                        st.session_state.chat_messages = []
                        st.success("System state reset!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to reset state: {e}")
            
            with action_col2:
                if st.button("⚡ Optimize System", type="secondary"):
                    with st.spinner("Optimizing system..."):
                        try:
                            # Optimize vectors if available
                            if vector_stats['vector_enabled'] and vector_stats.get('documents_with_embeddings', 0) > 0:
                                chatbot.vector.quantize_vectors()
                                chatbot.vector.preload_quantized_vectors()
                            
                            st.success("System optimization complete!")
                        except Exception as e:
                            st.error(f"Optimization failed: {e}")
    else:
        st.warning("⚠️ No model loaded. Please load a model in the Chat tab to view system information.")