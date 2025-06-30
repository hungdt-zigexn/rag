#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================================================
# Contextual Compression RAG (Phiên bản Ruby)
# ===============================================================================
#
# Contextual Compression RAG giải quyết vấn đề "Information Overload" và
# "Signal-to-Noise Ratio" trong traditional RAG systems. Thay vì truyền
# toàn bộ retrieved documents vào LLM, kỹ thuật này:
#
# 1. **Retrieval**: Lấy candidate documents như bình thường
# 2. **Compression**: Sử dụng LLM để compress và filter chỉ relevant parts
# 3. **Contextual Filtering**: Loại bỏ noise, giữ lại signal strong nhất
# 4. **Generation**: Generate response từ compressed, high-quality context
#
# Benefits:
# - Giảm token usage (cost optimization)
# - Tăng chất lượng context (signal-to-noise ratio)
# - Faster inference (ít tokens để process)
# - Better focus (LLM concentrate vào relevant info only)
# - Reduced hallucination (ít noise → ít confusion)
#
# Process Flow:
# Query → Retrieval → Compression Filter → Compressed Context → Final Generation

require 'net/http'
require 'uri'
require 'json'
require 'matrix'
require 'pdf-reader'

# ===============================================================================
# Thiết lập Môi trường
# ===============================================================================
puts "=== Contextual Compression RAG ==="

# Cấu hình
OPENAI_API_KEY = ENV['OPENAI_API_KEY']
BASE_URL = 'https://api.studio.nebius.com/v1/'
EMBEDDING_MODEL = 'BAAI/bge-en-icl'
CHAT_MODEL = 'meta-llama/Llama-3.3-70B-Instruct'
COMPRESSOR_MODEL = 'meta-llama/Llama-3.3-70B-Instruct'  # Có thể dùng model khác

# Kiểm tra API key
if OPENAI_API_KEY.nil? || OPENAI_API_KEY.empty?
  puts "Cảnh báo: Biến môi trường OPENAI_API_KEY chưa được đặt"
  exit 1
end

# ===============================================================================
# API Client Functions
# ===============================================================================

def make_api_request(endpoint, payload)
  # Thực hiện API request đến OpenAI-compatible endpoint.
  uri = URI("#{BASE_URL}#{endpoint}")
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = true

  request = Net::HTTP::Post.new(uri)
  request['Authorization'] = "Bearer #{OPENAI_API_KEY}"
  request['Content-Type'] = 'application/json'
  request.body = payload.to_json

  response = http.request(request)

  if response.code.to_i == 200
    JSON.parse(response.body)
  else
    puts "Lỗi API: #{response.code} - #{response.body}"
    nil
  end
end

def create_embeddings(text_list, model = EMBEDDING_MODEL)
  # Tạo embeddings cho danh sách văn bản.
  payload = {
    model: model,
    input: text_list
  }

  response = make_api_request('embeddings', payload)
  return [] unless response && response['data']

  response['data'].map { |item| item['embedding'] }
end

def generate_response(system_prompt, user_message, model = CHAT_MODEL)
  # Tạo phản hồi từ mô hình AI.
  payload = {
    model: model,
    temperature: 0.2,  # Low temperature for precise compression
    messages: [
      { role: 'system', content: system_prompt },
      { role: 'user', content: user_message }
    ]
  }

  response = make_api_request('chat/completions', payload)
  return nil unless response && response['choices'] && !response['choices'].empty?

  response['choices'][0]['message']['content']
end

# ===============================================================================
# Text Processing Utilities
# ===============================================================================

def extract_text_from_pdf(pdf_path)
  # Trích xuất văn bản từ file PDF.
  reader = PDF::Reader.new(pdf_path)
  all_text = ""

  reader.pages.each do |page|
    all_text += page.text + " "
  end

  all_text.strip
end

def chunk_text(text, chunk_size = 1000, overlap = 200)
  # Chia văn bản thành các chunk có overlap.
  chunks = []
  step_size = chunk_size - overlap

  (0...text.length).step(step_size) do |i|
    chunk = text[i, chunk_size]
    chunks << chunk unless chunk.empty?
  end

  chunks
end

# ===============================================================================
# Initial Retrieval (Same as Traditional RAG)
# ===============================================================================

def cosine_similarity(vec1, vec2)
  # Tính cosine similarity giữa hai vector.
  v1 = Vector[*vec1]
  v2 = Vector[*vec2]

  dot_product = v1.inner_product(v2)
  magnitude1 = Math.sqrt(v1.inner_product(v1))
  magnitude2 = Math.sqrt(v2.inner_product(v2))

  return 0.0 if magnitude1 == 0.0 || magnitude2 == 0.0

  dot_product / (magnitude1 * magnitude2)
end

def initial_retrieval(query, chunks, embeddings, top_k = 10)
  # Stage 1: Initial retrieval - lấy candidate documents.
  #
  # Lấy nhiều hơn final cần thiết để có material cho compression.
  # Ví dụ: cần 3 final documents → retrieve 10 candidates → compress về 3.
  #
  # Args:
  #   query (String): User query.
  #   chunks (Array): Document chunks.
  #   embeddings (Array): Chunk embeddings.
  #   top_k (Integer): Số candidates để retrieve.
  #
  # Returns:
  #   Array: Top-k candidate chunks với scores.
  puts "\n--- Stage 1: Initial Retrieval ---"
  puts "Query: #{query}"
  puts "Retrieving top #{top_k} candidates for compression..."

  # Tạo embedding cho query
  query_embeddings = create_embeddings([query])
  return [] if query_embeddings.empty?

  query_embedding = query_embeddings[0]

  # Tính similarity với tất cả chunks
  similarities = embeddings.each_with_index.map do |chunk_emb, i|
    similarity = cosine_similarity(query_embedding, chunk_emb)
    [i, similarity, chunks[i]]
  end

  # Sắp xếp và lấy top k
  top_candidates = similarities.sort_by { |_, score, _| -score }.first(top_k)

  puts "Retrieved candidates:"
  top_candidates.each_with_index do |(index, score, chunk), rank|
    preview = chunk[0..80] + (chunk.length > 80 ? "..." : "")
    puts "#{rank + 1}. Chunk #{index}: Score #{score.round(4)} - #{preview}"
  end

  top_candidates
end

# ===============================================================================
# Core: Contextual Compression
# ===============================================================================

def compress_document(query, document, compression_type = "extract_relevant", model = COMPRESSOR_MODEL)
  # Compress một document dựa trên query context.
  #
  # Đây là kỹ thuật cốt lõi của Contextual Compression RAG:
  # Sử dụng LLM để intelligent filtering và compression.
  #
  # Args:
  #   query (String): User query làm context cho compression.
  #   document (String): Document cần compress.
  #   compression_type (String): Loại compression ("extract_relevant", "summarize", "filter").
  #   model (String): LLM model để compression.
  #
  # Returns:
  #   Hash: Compressed result với metadata.

  compression_prompts = {
    "extract_relevant" => "
    Bạn là chuyên gia compression thông tin. Nhiệm vụ của bạn là trích xuất
    chỉ những phần RELEVANT với query từ document được cung cấp.

    NGUYÊN TẮC COMPRESSION:
    1. **Relevance First**: Chỉ giữ lại phần directly relevant với query
    2. **Preserve Context**: Giữ enough context để information có nghĩa
    3. **Remove Noise**: Loại bỏ tangential information, redundancy
    4. **Maintain Accuracy**: Không thay đổi ý nghĩa của original information
    5. **Concise but Complete**: Ngắn gọn nhưng đầy đủ thông tin cần thiết

    COMPRESSION STRATEGY:
    - Extract key sentences/phrases that answer the query
    - Keep supporting details nếu cần thiết cho context
    - Remove examples/anecdotes không directly relevant
    - Maintain logical flow của information
    - Preserve exact facts, numbers, technical terms

    FORMAT OUTPUT:
    Trả về chỉ compressed content, KHÔNG có meta-commentary hoặc explanation.
    Compressed content phải standalone readable và coherent.
    ",

    "summarize" => "
    Bạn là chuyên gia summarization. Tạo summary ngắn gọn của document
    với focus vào aspects relevant với query.

    SUMMARIZATION GUIDELINES:
    1. **Query-Focused**: Summary phải relevant với query
    2. **Key Points**: Capture main points và key insights
    3. **Factual Accuracy**: Preserve important facts và details
    4. **Coherent Structure**: Logical flow của ideas
    5. **Appropriate Length**: 30-50% của original length

    FORMAT: Trả về coherent summary, KHÔNG có bullet points hoặc lists.
    ",

    "filter" => "
    Bạn là chuyên gia filtering information. Lọc document để chỉ giữ lại
    sentences/paragraphs có high relevance với query.

    FILTERING CRITERIA:
    1. **Direct Relevance**: Sentence directly answers hoặc relates đến query
    2. **Supporting Information**: Context cần thiết để hiểu direct answers
    3. **Remove Redundancy**: Loại bỏ repeated information
    4. **Remove Tangents**: Loại bỏ off-topic discussions

    FORMAT: Trả về filtered sentences, giữ nguyên original wording.
    "
  }

  system_prompt = compression_prompts[compression_type] || compression_prompts["extract_relevant"]

  user_message = "
QUERY: #{query}

DOCUMENT TO COMPRESS:
#{document}

Compressed Result:
"

  puts "\nCompressing document (#{compression_type})..."
  puts "Original length: #{document.length} chars"

  compressed_result = generate_response(system_prompt, user_message, model)

  if compressed_result
    compression_ratio = (compressed_result.length.to_f / document.length * 100).round(1)

    puts "Compressed length: #{compressed_result.length} chars"
    puts "Compression ratio: #{compression_ratio}%"
    puts "Preview: #{compressed_result[0..150]}..."

    {
      original_text: document,
      compressed_text: compressed_result,
      original_length: document.length,
      compressed_length: compressed_result.length,
      compression_ratio: compression_ratio,
      compression_type: compression_type
    }
  else
    puts "❌ Compression failed"
    {
      original_text: document,
      compressed_text: document,  # Fallback to original
      original_length: document.length,
      compressed_length: document.length,
      compression_ratio: 100.0,
      compression_type: "failed"
    }
  end
end

def contextual_compression_pipeline(query, candidate_documents, compression_type = "extract_relevant",
                                   target_final_count = 5)
  # Complete contextual compression pipeline.
  #
  # Process multiple candidate documents → compress each → optionally re-rank
  # → return top compressed documents.
  #
  # Args:
  #   query (String): User query.
  #   candidate_documents (Array): Candidates từ initial retrieval.
  #   compression_type (String): Type of compression to apply.
  #   target_final_count (Integer): Final number of documents to return.
  #
  # Returns:
  #   Hash: Complete compression results.
  puts "\n--- Stage 2: Contextual Compression Pipeline ---"
  puts "Compressing #{candidate_documents.length} candidates → #{target_final_count} final docs"
  puts "Compression type: #{compression_type}"

  compressed_documents = []
  total_original_chars = 0
  total_compressed_chars = 0

  candidate_documents.each_with_index do |(chunk_idx, similarity_score, document), i|
    puts "\nProcessing candidate #{i + 1}/#{candidate_documents.length} (Chunk #{chunk_idx})"

    # Compress the document
    compression_result = compress_document(query, document, compression_type)

    # Track statistics
    total_original_chars += compression_result[:original_length]
    total_compressed_chars += compression_result[:compressed_length]

    # Store result với additional metadata
    compressed_documents << {
      chunk_index: chunk_idx,
      original_similarity_score: similarity_score,
      compression_result: compression_result,
      candidate_rank: i + 1
    }
  end

  # Calculate overall compression statistics
  overall_compression_ratio = (total_compressed_chars.to_f / total_original_chars * 100).round(1)

  puts "\n--- Compression Statistics ---"
  puts "Total original chars: #{total_original_chars}"
  puts "Total compressed chars: #{total_compressed_chars}"
  puts "Overall compression ratio: #{overall_compression_ratio}%"
  puts "Space saved: #{((100 - overall_compression_ratio)).round(1)}%"

  # Optional: Re-rank compressed documents based on their compressed content
  # (có thể skip nếu muốn giữ original order)
  final_documents = compressed_documents.first(target_final_count)

  puts "\nFinal compressed documents (Top #{target_final_count}):"
  final_documents.each_with_index do |doc, i|
    puts "#{i + 1}. Chunk #{doc[:chunk_index]}: #{doc[:compression_result][:compressed_length]} chars"
    puts "   Compression: #{doc[:compression_result][:compression_ratio]}%"
    puts "   Preview: #{doc[:compression_result][:compressed_text][0..100]}..."
  end

  {
    query: query,
    original_candidates: candidate_documents,
    compressed_documents: compressed_documents,
    final_documents: final_documents,
    compression_stats: {
      total_original_chars: total_original_chars,
      total_compressed_chars: total_compressed_chars,
      overall_compression_ratio: overall_compression_ratio,
      space_saved_percentage: (100 - overall_compression_ratio).round(1),
      documents_processed: candidate_documents.length,
      final_count: final_documents.length
    }
  }
end

# ===============================================================================
# Advanced Compression Strategies
# ===============================================================================

def multi_stage_compression(query, candidate_documents)
  # Multi-stage compression: Apply multiple compression techniques in sequence.
  #
  # Stage 1: Filter (remove completely irrelevant parts)
  # Stage 2: Extract (extract relevant sections)
  # Stage 3: Summarize (condense remaining content)
  puts "\n--- Multi-Stage Compression ---"

  multi_stage_results = []

  candidate_documents.each_with_index do |(chunk_idx, similarity_score, document), i|
    puts "\nMulti-stage compression for candidate #{i + 1}"

    # Stage 1: Filter
    stage1_result = compress_document(query, document, "filter")

    # Stage 2: Extract từ filtered content
    stage2_result = compress_document(query, stage1_result[:compressed_text], "extract_relevant")

    # Stage 3: Summarize nếu vẫn còn quá dài
    final_text = if stage2_result[:compressed_length] > 300
      stage3_result = compress_document(query, stage2_result[:compressed_text], "summarize")
      stage3_result[:compressed_text]
    else
      stage2_result[:compressed_text]
    end

    final_compression_ratio = (final_text.length.to_f / document.length * 100).round(1)

    puts "Final result: #{document.length} → #{final_text.length} chars (#{final_compression_ratio}%)"

    multi_stage_results << {
      chunk_index: chunk_idx,
      original_similarity_score: similarity_score,
      original_text: document,
      stage1_filtered: stage1_result[:compressed_text],
      stage2_extracted: stage2_result[:compressed_text],
      final_compressed: final_text,
      final_compression_ratio: final_compression_ratio
    }
  end

  multi_stage_results
end

def adaptive_compression(query, candidate_documents)
  # Adaptive compression: Choose compression strategy based on document characteristics.
  #
  # - Short documents: Use "filter" approach
  # - Medium documents: Use "extract_relevant" approach
  # - Long documents: Use "summarize" approach
  puts "\n--- Adaptive Compression ---"

  adaptive_results = []

  candidate_documents.each_with_index do |(chunk_idx, similarity_score, document), i|
    doc_length = document.length

    # Choose compression strategy based on length
    compression_type = case doc_length
                      when 0..500
                        "filter"  # Short docs: just filter
                      when 501..1500
                        "extract_relevant"  # Medium docs: extract relevant parts
                      else
                        "summarize"  # Long docs: summarize
                      end

    puts "\nAdaptive compression for candidate #{i + 1} (#{doc_length} chars → #{compression_type})"

    compression_result = compress_document(query, document, compression_type)

    adaptive_results << {
      chunk_index: chunk_idx,
      original_similarity_score: similarity_score,
      compression_result: compression_result,
      strategy_used: compression_type,
      doc_length_category: case doc_length
                          when 0..500 then "short"
                          when 501..1500 then "medium"
                          else "long"
                          end
    }
  end

  adaptive_results
end

# ===============================================================================
# Complete Contextual Compression RAG Pipeline
# ===============================================================================

def contextual_compression_rag(query, chunks, embeddings, retrieval_k = 10, final_k = 5,
                               compression_type = "extract_relevant")
  # Complete Contextual Compression RAG pipeline.
  #
  # Query → Initial Retrieval → Contextual Compression → Final Generation
  #
  # Args:
  #   query (String): User query.
  #   chunks (Array): Document chunks.
  #   embeddings (Array): Chunk embeddings.
  #   retrieval_k (Integer): Candidates to retrieve initially.
  #   final_k (Integer): Final documents after compression.
  #   compression_type (String): Type of compression.
  #
  # Returns:
  #   Hash: Complete pipeline results.
  puts "\n" + "=" * 80
  puts "CONTEXTUAL COMPRESSION RAG PIPELINE"
  puts "=" * 80
  puts "Query: #{query}"
  puts "Initial retrieval: #{retrieval_k} candidates"
  puts "Final after compression: #{final_k} documents"
  puts "Compression type: #{compression_type}"
  puts "=" * 80

  # Stage 1: Initial Retrieval
  candidates = initial_retrieval(query, chunks, embeddings, retrieval_k)
  return { error: "No candidates retrieved" } if candidates.empty?

  # Stage 2: Contextual Compression
  compression_results = contextual_compression_pipeline(query, candidates, compression_type, final_k)

  # Stage 3: Generate final response using compressed context
  compressed_context = compression_results[:final_documents]
                       .map.with_index { |doc, i| "CONTEXT #{i + 1}:\n#{doc[:compression_result][:compressed_text]}" }
                       .join("\n\n")

  puts "\n--- Stage 3: Final Generation ---"
  puts "Generating response với compressed context..."
  puts "Compressed context length: #{compressed_context.length} chars"

  generation_system_prompt = "
  Bạn là AI assistant với khả năng xử lý compressed context efficiently.
  Bạn sẽ nhận được context đã được compression để focus vào relevant information.

  HƯỚNG DẪN:
  1. Sử dụng compressed context để trả lời câu hỏi một cách chính xác
  2. Compressed context đã được filtered để loại bỏ noise
  3. Focus vào key information được preserved trong compression
  4. Nếu thông tin không đủ, nói rõ limitations
  5. Maintain accuracy và avoid hallucination
  "

  generation_user_message = "
#{compressed_context}

QUESTION: #{query}

ANSWER:
"

  final_response = generate_response(generation_system_prompt, generation_user_message, CHAT_MODEL)

  if final_response
    puts "✓ Generated response (#{final_response.length} chars)"
    puts "Preview: #{final_response[0..200]}..."
  else
    puts "❌ Failed to generate final response"
    final_response = "Unable to generate response from compressed context."
  end

  {
    query: query,
    retrieval_results: candidates,
    compression_results: compression_results,
    compressed_context: compressed_context,
    final_response: final_response,
    pipeline_stats: {
      candidates_retrieved: candidates.length,
      final_documents: final_k,
      total_compression_ratio: compression_results[:compression_stats][:overall_compression_ratio],
      context_length: compressed_context.length,
      response_length: final_response&.length || 0
    }
  }
end

# ===============================================================================
# Comparison Analysis
# ===============================================================================

def compare_traditional_vs_compression(query, chunks, embeddings, k = 5)
  """
  So sánh Traditional RAG vs Contextual Compression RAG.
  """
  puts "\n" + "=" * 80
  puts "COMPARISON: TRADITIONAL RAG vs CONTEXTUAL COMPRESSION RAG"
  puts "=" * 80

  # Traditional RAG
  puts "\n1. TRADITIONAL RAG:"
  puts "-" * 50
  traditional_candidates = initial_retrieval(query, chunks, embeddings, k)
  traditional_context = traditional_candidates.map.with_index do |(_, _, chunk), i|
    "CONTEXT #{i + 1}:\n#{chunk}"
  end.join("\n\n")

  puts "Traditional context length: #{traditional_context.length} chars"
  puts "Number of documents: #{traditional_candidates.length}"

  # Contextual Compression RAG
  puts "\n2. CONTEXTUAL COMPRESSION RAG:"
  puts "-" * 50
  compression_result = contextual_compression_rag(query, chunks, embeddings, k * 2, k, "extract_relevant")

  return unless compression_result[:compressed_context]

  # Comparison metrics
  puts "\n3. COMPARISON METRICS:"
  puts "-" * 50

  traditional_length = traditional_context.length
  compressed_length = compression_result[:compressed_context].length
  compression_ratio = (compressed_length.to_f / traditional_length * 100).round(1)

  puts "Traditional RAG:"
  puts "  Context length: #{traditional_length} chars"
  puts "  Documents used: #{traditional_candidates.length}"
  puts "  Processing: Full documents"

  puts "\nContextual Compression RAG:"
  puts "  Context length: #{compressed_length} chars"
  puts "  Documents used: #{compression_result[:pipeline_stats][:final_documents]}"
  puts "  Processing: Compressed documents"
  puts "  Compression ratio: #{compression_ratio}%"
  puts "  Space saved: #{(100 - compression_ratio).round(1)}%"

  # Generate responses với both approaches cho comparison
  puts "\n4. RESPONSE COMPARISON:"
  puts "-" * 50

  # Traditional response
  traditional_system_prompt = """
  Bạn là AI assistant. Trả lời câu hỏi dựa trên context được cung cấp.
  """

  traditional_user_message = """
#{traditional_context}

QUESTION: #{query}

ANSWER:
"""

  traditional_response = generate_response(traditional_system_prompt, traditional_user_message, CHAT_MODEL)

  puts "Traditional RAG Response:"
  puts "  Length: #{traditional_response&.length || 0} chars"
  puts "  Preview: #{traditional_response&.[](0..200) || 'Failed'}..."

  puts "\nContextual Compression RAG Response:"
  puts "  Length: #{compression_result[:final_response]&.length || 0} chars"
  puts "  Preview: #{compression_result[:final_response]&.[](0..200) || 'Failed'}..."

  # Quality metrics (basic)
  traditional_words = traditional_response&.split&.length || 0
  compressed_words = compression_result[:final_response]&.split&.length || 0

  puts "\n5. EFFICIENCY METRICS:"
  puts "-" * 50
  puts "Context efficiency: #{compression_ratio}% size for processing"
  puts "Response lengths: Traditional=#{traditional_words} words, Compressed=#{compressed_words} words"
  puts "Token savings: ~#{((traditional_length - compressed_length) / 4).round} tokens saved in context"

  {
    traditional: {
      context_length: traditional_length,
      response: traditional_response,
      documents_used: traditional_candidates.length
    },
    compressed: {
      context_length: compressed_length,
      response: compression_result[:final_response],
      documents_used: compression_result[:pipeline_stats][:final_documents],
      compression_ratio: compression_ratio
    },
    efficiency_gains: {
      space_saved_percentage: (100 - compression_ratio).round(1),
      estimated_token_savings: ((traditional_length - compressed_length) / 4).round
    }
  }
end

# ===============================================================================
# Demo chính
# ===============================================================================

def run_contextual_compression_demo
  """
  Chạy demo hoàn chỉnh Contextual Compression RAG.
  """
  puts "\n=== Demo Contextual Compression RAG ==="

  # Bước 1: Load và chuẩn bị data
  puts "\n1. Chuẩn bị dữ liệu..."
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  chunks = chunk_text(extracted_text, 1200, 200)

  # Giới hạn chunks cho demo
  demo_chunks = chunks.first(12)
  puts "Demo với #{demo_chunks.length} chunks"

  # Tạo embeddings
  puts "Tạo embeddings cho chunks..."
  embeddings = create_embeddings(demo_chunks)
  return unless embeddings.any?

  # Bước 2: Test với multiple queries
  test_queries = [
    "What are the practical applications of AI in healthcare and medicine?",
    "How can businesses implement machine learning to improve efficiency?",
    "What are the main challenges and risks of artificial intelligence?",
    "Explain the benefits of AI technology for different industries",
    "How does AI impact employment and what are the future implications?"
  ]

  # Bước 3: Run comparison cho mỗi query
  test_queries.each_with_index do |query, i|
    puts "\n" + "=" * 100
    puts "DEMO #{i + 1}: Contextual Compression Analysis"
    puts "=" * 100

    # Compare traditional vs compression
    comparison = compare_traditional_vs_compression(query, demo_chunks, embeddings, 5)

    puts "\n" + "=" * 100
  end

  # Bước 4: Demo advanced compression strategies
  puts "\n" + "=" * 100
  puts "ADVANCED COMPRESSION STRATEGIES DEMO"
  puts "=" * 100

  demo_query = test_queries[0]
  demo_candidates = initial_retrieval(demo_query, demo_chunks, embeddings, 6)

  # Multi-stage compression
  puts "\nTesting Multi-Stage Compression:"
  multi_stage_results = multi_stage_compression(demo_query, demo_candidates.first(3))

  # Adaptive compression
  puts "\nTesting Adaptive Compression:"
  adaptive_results = adaptive_compression(demo_query, demo_candidates.first(3))

  puts "\nAdvanced strategies completed!"
end

# ===============================================================================
# Compression Quality Analysis
# ===============================================================================

def analyze_compression_quality
  """
  Phân tích chất lượng của various compression strategies.
  """
  puts "\n=== Phân tích chất lượng Compression ==="

  # Load data
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  chunks = chunk_text(extracted_text, 1000, 150).first(5)
  embeddings = create_embeddings(chunks)
  return unless embeddings.any?

  test_query = "How does artificial intelligence impact healthcare?"
  candidates = initial_retrieval(test_query, chunks, embeddings, 5)

  compression_types = ["filter", "extract_relevant", "summarize"]

  puts "Testing compression quality across different strategies:"
  puts "=" * 60

  quality_results = {}

  compression_types.each do |comp_type|
    puts "\n#{comp_type.upcase} COMPRESSION:"
    puts "-" * 30

    total_original = 0
    total_compressed = 0
    compression_times = []

    candidates.first(3).each_with_index do |(chunk_idx, score, document), i|
      start_time = Time.now
      result = compress_document(test_query, document, comp_type)
      compression_time = Time.now - start_time

      total_original += result[:original_length]
      total_compressed += result[:compressed_length]
      compression_times << compression_time

      puts "Document #{i + 1}: #{result[:original_length]} → #{result[:compressed_length]} chars (#{result[:compression_ratio]}%)"
    end

    avg_compression_ratio = (total_compressed.to_f / total_original * 100).round(1)
    avg_compression_time = (compression_times.sum / compression_times.length).round(2)

    quality_results[comp_type] = {
      avg_compression_ratio: avg_compression_ratio,
      avg_compression_time: avg_compression_time,
      total_space_saved: (100 - avg_compression_ratio).round(1)
    }

    puts "Average compression ratio: #{avg_compression_ratio}%"
    puts "Average compression time: #{avg_compression_time}s"
    puts "Space saved: #{quality_results[comp_type][:total_space_saved]}%"
  end

  # Summary comparison
  puts "\n" + "=" * 60
  puts "COMPRESSION STRATEGY COMPARISON"
  puts "=" * 60

  quality_results.each do |strategy, metrics|
    puts "#{strategy.upcase}:"
    puts "  Compression: #{metrics[:avg_compression_ratio]}%"
    puts "  Space saved: #{metrics[:total_space_saved]}%"
    puts "  Avg time: #{metrics[:avg_compression_time]}s"
  end

  # Best strategy recommendation
  most_efficient = quality_results.min_by { |_, metrics| metrics[:avg_compression_ratio] }
  fastest = quality_results.min_by { |_, metrics| metrics[:avg_compression_time] }

  puts "\nRecommendations:"
  puts "  Most efficient (highest compression): #{most_efficient[0].upcase}"
  puts "  Fastest processing: #{fastest[0].upcase}"
end

# ===============================================================================
# Main Execution
# ===============================================================================

if __FILE__ == $0
  puts "Contextual Compression RAG bằng Ruby"
  puts "=" * 60

  begin
    # Demo chính
    run_contextual_compression_demo

    # Phân tích compression quality
    puts "\n\n"
    analyze_compression_quality

    puts "\n=== HOÀN THÀNH DEMO CONTEXTUAL COMPRESSION RAG ==="
  rescue => e
    puts "Lỗi trong quá trình thực thi: #{e.message}"
    puts e.backtrace
  end
end
