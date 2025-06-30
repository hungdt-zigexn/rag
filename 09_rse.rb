#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================================================
# RSE - Relevant Segment Extraction RAG (Phiên bản Ruby)
# ===============================================================================
#
# RSE (Relevant Segment Extraction) giải quyết vấn đề "Fragmented Information"
# trong traditional chunk-based retrieval. Thay vì trả về isolated chunks,
# RSE tìm kiếm và trả về continuous segments của text.
#
# Vấn đề với Chunk-based Retrieval:
# - Chunks độc lập có thể thiếu context
# - Information có thể bị split across multiple chunks
# - Khó maintain coherence và flow của original text
#
# RSE Approach:
# 1. Chia text thành overlapping windows (sliding window)
# 2. Score mỗi window dựa trên relevance với query
# 3. Identify continuous segments với high relevance scores
# 4. Merge adjacent high-scoring windows thành coherent segments
# 5. Return segments với preserved context và natural boundaries
#
# Kỹ thuật chính: Sliding Window Relevance Scoring + Segment Merging

require 'net/http'
require 'uri'
require 'json'
require 'matrix'
require 'pdf-reader'

# ===============================================================================
# Thiết lập Môi trường
# ===============================================================================
puts "=== RSE - Relevant Segment Extraction RAG ==="

# Cấu hình
OPENAI_API_KEY = ENV['OPENAI_API_KEY']
BASE_URL = 'https://api.studio.nebius.com/v1/'
EMBEDDING_MODEL = 'BAAI/bge-en-icl'
CHAT_MODEL = 'meta-llama/Llama-3.3-70B-Instruct'

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
    temperature: 0.3,
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

def create_sliding_windows(text, window_size = 500, step_size = 250)
  # Tạo sliding windows từ text.
  #
  # Khác với traditional chunking, sliding windows có overlap cao
  # để đảm bảo không mất thông tin quan trọng giữa boundaries.
  #
  # Args:
  #   text (String): Văn bản đầu vào.
  #   window_size (Integer): Kích thước mỗi window (characters).
  #   step_size (Integer): Khoảng cách giữa start positions của các windows.
  #
  # Returns:
  #   Array: Mảng các windows với metadata.
  puts "\n--- Tạo Sliding Windows ---"
  puts "Window size: #{window_size}, Step size: #{step_size}"

  windows = []
  position = 0

  while position < text.length
    # Extract window content
    window_end = [position + window_size, text.length].min
    window_text = text[position...window_end]

    # Skip nếu window quá ngắn
    next if window_text.length < 100

    # Tìm natural boundary (end of sentence) nếu có thể
    if window_end < text.length
      # Tìm sentence end gần boundary
      last_period = window_text.rindex('.')
      last_question = window_text.rindex('?')
      last_exclamation = window_text.rindex('!')

      natural_end = [last_period, last_question, last_exclamation].compact.max

      # Nếu tìm được natural boundary trong 50 chars cuối
      if natural_end && natural_end > window_text.length - 50
        window_text = window_text[0..natural_end]
      end
    end

    windows << {
      id: windows.length,
      start_pos: position,
      end_pos: position + window_text.length,
      text: window_text,
      length: window_text.length
    }

    position += step_size
  end

  puts "Tạo được #{windows.length} windows"
  puts "Trung bình độ dài window: #{(windows.map { |w| w[:length] }.sum.to_f / windows.length).round(1)} chars"

  windows
end

# ===============================================================================
# Core RSE Algorithm: Sliding Window Relevance Scoring
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

def score_windows_for_relevance(query, windows)
  # Score từng window dựa trên relevance với query.
  #
  # Đây là bước cốt lõi của RSE - tính relevance score cho mỗi
  # sliding window để identify những segments có chứa thông tin relevant.
  #
  # Args:
  #   query (String): User query.
  #   windows (Array): Sliding windows.
  #
  # Returns:
  #   Array: Windows với relevance scores.
  puts "\n--- Scoring Windows for Relevance ---"
  puts "Query: #{query}"
  puts "Scoring #{windows.length} windows..."

  # Tạo embedding cho query
  query_embeddings = create_embeddings([query])
  return [] if query_embeddings.empty?

  query_embedding = query_embeddings[0]

  # Tạo embeddings cho tất cả windows (batch processing for efficiency)
  window_texts = windows.map { |w| w[:text] }
  window_embeddings = create_embeddings(window_texts)

  return [] if window_embeddings.empty?

  # Tính relevance scores
  scored_windows = windows.each_with_index.map do |window, i|
    window_embedding = window_embeddings[i]
    relevance_score = cosine_similarity(query_embedding, window_embedding)

    window.merge({
      relevance_score: relevance_score,
      embedding: window_embedding
    })
  end

  # Sort theo relevance score để analysis
  sorted_windows = scored_windows.sort_by { |w| -w[:relevance_score] }

  puts "Top 5 windows theo relevance score:"
  sorted_windows.first(5).each_with_index do |window, i|
    puts "#{i + 1}. Window #{window[:id]}: Score #{window[:relevance_score].round(4)}"
    puts "   Preview: #{window[:text][0..100]}..."
  end

  scored_windows
end

def identify_relevant_segments(scored_windows, score_threshold = 0.3, min_segment_length = 200)
  # Identify continuous segments có high relevance scores.
  #
  # Đây là bước quan trọng của RSE: thay vì trả về isolated windows,
  # tìm kiếm những continuous segments (adjacent windows với high scores)
  # để maintain context và coherence.
  #
  # Args:
  #   scored_windows (Array): Windows với relevance scores.
  #   score_threshold (Float): Minimum score để consider "relevant".
  #   min_segment_length (Integer): Minimum length cho valid segment.
  #
  # Returns:
  #   Array: Relevant segments với merged content.
  puts "\n--- Identifying Relevant Segments ---"
  puts "Score threshold: #{score_threshold}"
  puts "Min segment length: #{min_segment_length}"

  # Sort windows theo position để identify continuity
  windows_by_position = scored_windows.sort_by { |w| w[:start_pos] }

  # Identify relevant windows (above threshold)
  relevant_windows = windows_by_position.select { |w| w[:relevance_score] >= score_threshold }

  puts "Found #{relevant_windows.length} windows above threshold"

  return [] if relevant_windows.empty?

  # Group adjacent relevant windows into segments
  segments = []
  current_segment = [relevant_windows.first]

  relevant_windows[1..-1].each do |window|
    last_window = current_segment.last

    # Check if this window is adjacent to last window in current segment
    # Adjacent = overlapping hoặc very close positions
    overlap_start = [last_window[:start_pos], window[:start_pos]].max
    overlap_end = [last_window[:end_pos], window[:end_pos]].min

    is_adjacent = if overlap_start <= overlap_end
      # Overlapping windows
      true
    else
      # Non-overlapping but close windows (gap < 100 chars)
      (window[:start_pos] - last_window[:end_pos]) < 100
    end

    if is_adjacent
      current_segment << window
    else
      # Finalize current segment và start new one
      segments << current_segment if current_segment.any?
      current_segment = [window]
    end
  end

  # Add last segment
  segments << current_segment if current_segment.any?

  puts "Grouped windows into #{segments.length} segments"

  # Merge windows trong mỗi segment thành continuous text
  merged_segments = segments.map.with_index do |segment_windows, i|
    # Sort windows in segment by position
    sorted_segment = segment_windows.sort_by { |w| w[:start_pos] }

    # Calculate segment bounds
    segment_start = sorted_segment.first[:start_pos]
    segment_end = sorted_segment.last[:end_pos]

    # Merge text with overlap handling
    merged_text = ""
    last_end_pos = segment_start

    sorted_segment.each do |window|
      window_start = window[:start_pos]
      window_text = window[:text]

      if window_start >= last_end_pos
        # No overlap, append full text
        merged_text += " " if merged_text.length > 0
        merged_text += window_text
      else
        # Handle overlap by skipping overlapped part
        overlap_chars = last_end_pos - window_start
        if overlap_chars < window_text.length
          non_overlap_text = window_text[overlap_chars..-1]
          merged_text += non_overlap_text
        end
      end

      last_end_pos = window[:end_pos]
    end

    # Calculate average relevance score for segment
    avg_relevance = (segment_windows.map { |w| w[:relevance_score] }.sum / segment_windows.length)
    max_relevance = segment_windows.map { |w| w[:relevance_score] }.max

    segment_info = {
      id: i,
      start_pos: segment_start,
      end_pos: segment_end,
      text: merged_text.strip,
      length: merged_text.strip.length,
      num_windows: segment_windows.length,
      avg_relevance_score: avg_relevance,
      max_relevance_score: max_relevance,
      window_ids: segment_windows.map { |w| w[:id] }
    }

    puts "Segment #{i + 1}: #{segment_info[:length]} chars, #{segment_info[:num_windows]} windows, score #{avg_relevance.round(4)}"

    segment_info
  end

  # Filter segments theo min_segment_length
  valid_segments = merged_segments.select { |s| s[:length] >= min_segment_length }

  puts "Final segments after filtering: #{valid_segments.length}"

  # Sort by relevance score
  valid_segments.sort_by { |s| -s[:max_relevance_score] }
end

# ===============================================================================
# Complete RSE Pipeline
# ===============================================================================

def relevant_segment_extraction(query, text, window_size = 500, step_size = 250,
                                score_threshold = 0.3, max_segments = 5)
  # Complete RSE pipeline: từ text → sliding windows → relevance scoring → segment extraction.
  #
  # Args:
  #   query (String): User query.
  #   text (String): Full document text.
  #   window_size (Integer): Size của sliding windows.
  #   step_size (Integer): Step size cho sliding windows.
  #   score_threshold (Float): Threshold cho relevant windows.
  #   max_segments (Integer): Maximum segments to return.
  #
  # Returns:
  #   Hash: Complete RSE results.
  puts "\n" + "=" * 80
  puts "RSE - RELEVANT SEGMENT EXTRACTION PIPELINE"
  puts "=" * 80
  puts "Query: #{query}"
  puts "Text length: #{text.length} characters"
  puts "Window size: #{window_size}, Step size: #{step_size}"
  puts "Score threshold: #{score_threshold}"
  puts "=" * 80

  # Step 1: Create sliding windows
  windows = create_sliding_windows(text, window_size, step_size)
  return { error: "No windows created" } if windows.empty?

  # Step 2: Score windows for relevance
  scored_windows = score_windows_for_relevance(query, windows)
  return { error: "Failed to score windows" } if scored_windows.empty?

  # Step 3: Identify và merge relevant segments
  segments = identify_relevant_segments(scored_windows, score_threshold)

  # Step 4: Select top segments
  top_segments = segments.first([max_segments, segments.length].min)

  puts "\n--- RSE Results Summary ---"
  puts "Total windows created: #{windows.length}"
  puts "Windows above threshold: #{scored_windows.count { |w| w[:relevance_score] >= score_threshold }}"
  puts "Segments identified: #{segments.length}"
  puts "Top segments returned: #{top_segments.length}"

  if top_segments.any?
    puts "\nTop Segments:"
    top_segments.each_with_index do |segment, i|
      puts "#{i + 1}. Score: #{segment[:max_relevance_score].round(4)}, Length: #{segment[:length]} chars"
      puts "   Preview: #{segment[:text][0..150]}..."
    end
  end

  {
    query: query,
    total_windows: windows.length,
    scored_windows: scored_windows,
    identified_segments: segments,
    top_segments: top_segments,
    extraction_stats: {
      above_threshold_count: scored_windows.count { |w| w[:relevance_score] >= score_threshold },
      threshold_used: score_threshold,
      avg_window_score: (scored_windows.map { |w| w[:relevance_score] }.sum / scored_windows.length).round(4),
      max_window_score: scored_windows.map { |w| w[:relevance_score] }.max.round(4)
    }
  }
end

# ===============================================================================
# Comparison với Traditional Chunking
# ===============================================================================

def traditional_chunking(text, chunk_size = 1000, overlap = 200)
  """
  Traditional chunking để so sánh với RSE.
  """
  chunks = []
  step_size = chunk_size - overlap

  (0...text.length).step(step_size) do |i|
    chunk = text[i, chunk_size]
    chunks << {
      id: chunks.length,
      start_pos: i,
      end_pos: i + chunk.length,
      text: chunk,
      length: chunk.length
    } unless chunk.empty?
  end

  chunks
end

def compare_rse_vs_chunking(query, text)
  """
  So sánh RSE với traditional chunking approach.
  """
  puts "\n" + "=" * 80
  puts "COMPARISON: RSE vs TRADITIONAL CHUNKING"
  puts "=" * 80

  # RSE Approach
  puts "\n1. RSE APPROACH:"
  puts "-" * 50
  rse_results = relevant_segment_extraction(query, text, 500, 250, 0.3, 5)

  # Traditional Chunking Approach
  puts "\n2. TRADITIONAL CHUNKING APPROACH:"
  puts "-" * 50
  chunks = traditional_chunking(text, 1000, 200)
  puts "Created #{chunks.length} traditional chunks"

  # Score chunks với same query
  puts "Scoring chunks for relevance..."
  chunk_texts = chunks.map { |c| c[:text] }

  query_embeddings = create_embeddings([query])
  return unless query_embeddings.any?

  chunk_embeddings = create_embeddings(chunk_texts)
  return unless chunk_embeddings.any?

  query_embedding = query_embeddings[0]

  scored_chunks = chunks.each_with_index.map do |chunk, i|
    chunk_embedding = chunk_embeddings[i]
    relevance_score = cosine_similarity(query_embedding, chunk_embedding)
    chunk.merge(relevance_score: relevance_score)
  end

  top_chunks = scored_chunks.sort_by { |c| -c[:relevance_score] }.first(5)

  puts "Top 5 chunks:"
  top_chunks.each_with_index do |chunk, i|
    puts "#{i + 1}. Score: #{chunk[:relevance_score].round(4)}, Length: #{chunk[:length]} chars"
    puts "   Preview: #{chunk[:text][0..150]}..."
  end

  # Comparison Analysis
  puts "\n3. DETAILED COMPARISON:"
  puts "-" * 50

  rse_top = rse_results[:top_segments].first
  chunk_top = top_chunks.first

  if rse_top && chunk_top
    puts "Best RSE Segment:"
    puts "  Score: #{rse_top[:max_relevance_score].round(4)}"
    puts "  Length: #{rse_top[:length]} chars"
    puts "  Windows merged: #{rse_top[:num_windows]}"
    puts "  Content: #{rse_top[:text][0..200]}..."

    puts "\nBest Traditional Chunk:"
    puts "  Score: #{chunk_top[:relevance_score].round(4)}"
    puts "  Length: #{chunk_top[:length]} chars"
    puts "  Content: #{chunk_top[:text][0..200]}..."

    puts "\nComparison Metrics:"
    puts "  RSE vs Chunk score: #{rse_top[:max_relevance_score].round(4)} vs #{chunk_top[:relevance_score].round(4)}"
    score_improvement = ((rse_top[:max_relevance_score] - chunk_top[:relevance_score]) / chunk_top[:relevance_score] * 100).round(1)
    puts "  Score improvement: #{score_improvement}%"

    # Content overlap analysis
    rse_words = rse_top[:text].downcase.split
    chunk_words = chunk_top[:text].downcase.split
    common_words = (rse_words & chunk_words).length
    overlap_percentage = (common_words.to_f / [rse_words.length, chunk_words.length].min * 100).round(1)

    puts "  Content overlap: #{overlap_percentage}%"
  end

  # Statistics comparison
  puts "\n4. STATISTICS COMPARISON:"
  puts "-" * 50

  puts "RSE Stats:"
  puts "  Total elements: #{rse_results[:total_windows]} windows → #{rse_results[:identified_segments].length} segments"
  puts "  Average segment score: #{rse_results[:top_segments].map { |s| s[:max_relevance_score] }.sum / rse_results[:top_segments].length if rse_results[:top_segments].any?}"
  puts "  Coverage approach: Sliding windows với overlap cao"

  puts "\nTraditional Chunking Stats:"
  puts "  Total elements: #{chunks.length} chunks"
  puts "  Average chunk score: #{(scored_chunks.map { |c| c[:relevance_score] }.sum / scored_chunks.length).round(4)}"
  puts "  Coverage approach: Fixed-size chunks với overlap thấp"

  {
    rse_results: rse_results,
    chunk_results: scored_chunks,
    comparison: {
      rse_best_score: rse_top&.dig(:max_relevance_score),
      chunk_best_score: chunk_top&.dig(:relevance_score),
      score_improvement: score_improvement
    }
  }
end

# ===============================================================================
# Advanced RSE Analysis
# ===============================================================================

def analyze_rse_parameter_effects(query, text)
  """
  Phân tích ảnh hưởng của các parameters khác nhau lên RSE performance.
  """
  puts "\n=== Phân tích RSE Parameter Effects ==="

  # Test với different window sizes
  window_sizes = [300, 500, 700]
  step_ratios = [0.3, 0.5, 0.7]  # step_size as ratio of window_size
  threshold_values = [0.2, 0.3, 0.4]

  results = {}

  puts "Testing different parameter combinations..."
  puts "=" * 60

  window_sizes.each do |window_size|
    step_ratios.each do |step_ratio|
      threshold_values.each do |threshold|
        step_size = (window_size * step_ratio).to_i

        puts "\nTesting: Window=#{window_size}, Step=#{step_size}, Threshold=#{threshold}"

        begin
          rse_result = relevant_segment_extraction(query, text, window_size, step_size, threshold, 3)

          key = "W#{window_size}_S#{step_size}_T#{threshold}"
          results[key] = {
            params: { window_size: window_size, step_size: step_size, threshold: threshold },
            segments_found: rse_result[:identified_segments].length,
            avg_score: rse_result[:top_segments].any? ?
                      (rse_result[:top_segments].map { |s| s[:max_relevance_score] }.sum / rse_result[:top_segments].length).round(4) : 0,
            total_length: rse_result[:top_segments].sum { |s| s[:length] },
            windows_created: rse_result[:total_windows]
          }

          puts "  → Segments: #{results[key][:segments_found]}, Avg Score: #{results[key][:avg_score]}"
        rescue => e
          puts "  → Error: #{e.message}"
        end
      end
    end
  end

  # Analysis summary
  puts "\n" + "=" * 60
  puts "PARAMETER ANALYSIS SUMMARY"
  puts "=" * 60

  # Best configuration by average score
  best_config = results.max_by { |_, data| data[:avg_score] }
  if best_config
    puts "Best configuration by avg score:"
    params = best_config[1][:params]
    puts "  Window: #{params[:window_size]}, Step: #{params[:step_size]}, Threshold: #{params[:threshold]}"
    puts "  Avg Score: #{best_config[1][:avg_score]}, Segments: #{best_config[1][:segments_found]}"
  end

  # Best configuration by segment count
  best_count = results.max_by { |_, data| data[:segments_found] }
  if best_count
    puts "\nBest configuration by segment count:"
    params = best_count[1][:params]
    puts "  Window: #{params[:window_size]}, Step: #{params[:step_size]}, Threshold: #{params[:threshold]}"
    puts "  Segments: #{best_count[1][:segments_found]}, Avg Score: #{best_count[1][:avg_score]}"
  end

  results
end

# ===============================================================================
# Demo chính
# ===============================================================================

def run_rse_demo
  """
  Chạy demo hoàn chỉnh RSE với multiple test cases.
  """
  puts "\n=== Demo RSE - Relevant Segment Extraction ==="

  # Bước 1: Load và chuẩn bị data
  puts "\n1. Chuẩn bị dữ liệu..."
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)

  # Sử dụng subset của text cho demo (RSE works better với longer texts)
  demo_text = extracted_text[0..15000]  # First 15k characters
  puts "Demo text length: #{demo_text.length} characters"

  # Bước 2: Test với multiple queries
  test_queries = [
    "Các ứng dụng thực tế của AI trong chăm sóc sức khỏe là gì?",
    "Machine learning cải thiện hoạt động kinh doanh như thế nào?",
    "Những thách thức khi triển khai trí tuệ nhân tạo là gì?",
    "Giải thích lợi ích và rủi ro của công nghệ AI",
    "AI tác động đến việc làm và thị trường lao động như thế nào?"
  ]

  # Bước 3: Run RSE cho mỗi query
  test_queries.each_with_index do |query, i|
    puts "\n" + "=" * 100
    puts "DEMO #{i + 1}: RSE Analysis"
    puts "=" * 100

    # Run RSE
    rse_result = relevant_segment_extraction(query, demo_text, 500, 250, 0.3, 5)

    # Compare với traditional chunking
    compare_rse_vs_chunking(query, demo_text)

    puts "\n" + "=" * 100
  end

  # Bước 4: Parameter analysis với one representative query
  puts "\n" + "=" * 100
  puts "PARAMETER ANALYSIS"
  puts "=" * 100

  representative_query = test_queries[0]
  analyze_rse_parameter_effects(representative_query, demo_text)
end

# ===============================================================================
# Specialized RSE Functions
# ===============================================================================

def extract_context_around_segments(text, segments, context_chars = 200)
  """
  Extract additional context around identified segments.
  Hữu ích khi muốn expand segments để có more complete information.
  """
  puts "\n--- Extracting Context Around Segments ---"

  expanded_segments = segments.map do |segment|
    # Calculate expanded boundaries
    expanded_start = [segment[:start_pos] - context_chars, 0].max
    expanded_end = [segment[:end_pos] + context_chars, text.length].min

    # Extract expanded text
    expanded_text = text[expanded_start...expanded_end]

    # Find natural boundaries
    if expanded_start > 0
      # Find sentence start
      sentence_start = expanded_text.index(/[.!?]\s+[A-Z]/)
      if sentence_start
        expanded_text = expanded_text[sentence_start + 2..-1]
        expanded_start += sentence_start + 2
      end
    end

    if expanded_end < text.length
      # Find sentence end
      sentence_end = expanded_text.rindex(/[.!?]/)
      if sentence_end
        expanded_text = expanded_text[0..sentence_end]
        expanded_end = expanded_start + sentence_end + 1
      end
    end

    segment.merge({
      expanded_text: expanded_text,
      expanded_start: expanded_start,
      expanded_end: expanded_end,
      context_added: expanded_text.length - segment[:length]
    })
  end

  puts "Added context to #{expanded_segments.length} segments"
  expanded_segments.each_with_index do |seg, i|
    puts "Segment #{i + 1}: +#{seg[:context_added]} chars context"
  end

  expanded_segments
end

# ===============================================================================
# Main Execution
# ===============================================================================

if __FILE__ == $0
  puts "RSE - Relevant Segment Extraction RAG bằng Ruby"
  puts "=" * 60

  begin
    # Demo chính
    run_rse_demo

    puts "\n=== HOÀN THÀNH DEMO RSE RAG ==="
  rescue => e
    puts "Lỗi trong quá trình thực thi: #{e.message}"
    puts e.backtrace
  end
end
