#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================================================
# Context-Enriched RAG (Phiên bản Ruby)
# ===============================================================================
#
# Context-Enriched RAG giải quyết vấn đề "Isolated Chunks" - khi các chunk
# riêng lẻ thiếu ngữ cảnh xung quanh. Kỹ thuật này mở rộng bối cảnh bằng cách
# bao gồm các chunk lân cận để cung cấp thông tin đầy đủ hơn.
#
# Quy trình:
# 1. Trích xuất văn bản từ PDF.
# 2. Chia thành các chunk nhỏ.
# 3. Tạo embeddings cho từng chunk.
# 4. Tìm chunk có similarity cao nhất với query.
# 5. Mở rộng kết quả bằng cách lấy cả chunks xung quanh.
# 6. Sử dụng context mở rộng để tạo phản hồi.

require 'net/http'
require 'uri'
require 'json'
require 'matrix'
require 'pdf-reader'

# ===============================================================================
# Thiết lập Môi trường
# ===============================================================================
puts "=== Context-Enriched RAG ==="

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
# Hàm Trích xuất và Chia nhỏ
# ===============================================================================

def extract_text_from_pdf(pdf_path)
  # Trích xuất văn bản từ file PDF.
  #
  # Args:
  #   pdf_path (String): Đường dẫn đến file PDF.
  #
  # Returns:
  #   String: Văn bản được trích xuất từ PDF.
  reader = PDF::Reader.new(pdf_path)
  all_text = ""

  reader.pages.each do |page|
    all_text += page.text + " "
  end

  all_text.strip
end

def chunk_text(text, chunk_size = 1000, overlap = 200)
  # Chia văn bản thành các chunk có overlap.
  #
  # Args:
  #   text (String): Văn bản cần chia nhỏ.
  #   chunk_size (Integer): Số ký tự trong mỗi chunk.
  #   overlap (Integer): Số ký tự overlap giữa các chunk.
  #
  # Returns:
  #   Array<String>: Mảng các chunk văn bản.
  chunks = []
  step_size = chunk_size - overlap

  (0...text.length).step(step_size) do |i|
    chunk = text[i, chunk_size]
    chunks << chunk unless chunk.empty?
  end

  chunks
end

# ===============================================================================
# Hàm API Client
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
  #
  # Args:
  #   text_list (Array): Mảng văn bản đầu vào.
  #   model (String): Mô hình embedding.
  #
  # Returns:
  #   Array: Mảng embeddings.
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
    temperature: 0,
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
# Hàm Tìm kiếm Semantic
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

def basic_similarity_search(query, text_chunks, embeddings, k = 1)
  # Tìm kiếm similarity cơ bản - trả về top-k chunks có điểm cao nhất.
  #
  # Args:
  #   query (String): Câu hỏi của người dùng.
  #   text_chunks (Array): Mảng các text chunks.
  #   embeddings (Array): Embeddings của các chunks.
  #   k (Integer): Số lượng chunks cần trả về.
  #
  # Returns:
  #   Array: Các chunks liên quan nhất.
  # Tạo embedding cho query
  query_embeddings = create_embeddings([query])
  return [] if query_embeddings.empty?

  query_embedding = query_embeddings[0]

  # Tính similarity scores
  similarities = embeddings.each_with_index.map do |chunk_emb, i|
    similarity = cosine_similarity(query_embedding, chunk_emb)
    [i, similarity]
  end

  # Sắp xếp và lấy top k
  top_indices = similarities.sort_by { |_, score| -score }.first(k).map(&:first)
  top_indices.map { |i| text_chunks[i] }
end

# ===============================================================================
# Context-Enriched Search - Kỹ thuật Chính
# ===============================================================================

def context_enriched_search(query, text_chunks, embeddings, k = 1, context_size = 1)
  # Context-Enriched Search - Kỹ thuật cốt lõi của bài này.
  #
  # Thay vì chỉ trả về chunk có similarity cao nhất, chúng ta mở rộng
  # kết quả bằng cách bao gồm cả các chunk xung quanh để cung cấp
  # ngữ cảnh đầy đủ hơn.
  #
  # Args:
  #   query (String): Câu hỏi của người dùng.
  #   text_chunks (Array): Mảng các text chunks.
  #   embeddings (Array): Embeddings của các chunks.
  #   k (Integer): Số lượng chunks ban đầu cần tìm.
  #   context_size (Integer): Số chunks xung quanh để bao gồm (mỗi bên).
  #
  # Returns:
  #   Hash: {
  #     enhanced_chunks: Array - Chunks đã được mở rộng context,
  #     original_indices: Array - Index của chunks gốc,
  #     context_info: Hash - Thông tin về việc mở rộng context
  #   }
  puts "\n--- Context-Enriched Search đang hoạt động ---"

  # Bước 1: Tìm chunk có similarity cao nhất bằng basic search
  query_embeddings = create_embeddings([query])
  return { enhanced_chunks: [], original_indices: [], context_info: {} } if query_embeddings.empty?

  query_embedding = query_embeddings[0]

  # Tính similarity scores cho tất cả chunks
  similarities = embeddings.each_with_index.map do |chunk_emb, i|
    similarity = cosine_similarity(query_embedding, chunk_emb)
    [i, similarity]
  end

  # Lấy top chunk có điểm cao nhất
  top_indices = similarities.sort_by { |_, score| -score }.first(k).map(&:first)
  puts "Chunk có similarity cao nhất: index #{top_indices[0]} với điểm #{similarities.sort_by { |_, score| -score }.first[1].round(4)}"

  # Bước 2: Mở rộng context cho mỗi top chunk
  enhanced_chunks = []
  context_info = {}

  top_indices.each do |top_index|
    # Tính toán range để mở rộng context
    start_index = [0, top_index - context_size].max
    end_index = [text_chunks.length - 1, top_index + context_size].min

    puts "Mở rộng context từ index #{start_index} đến #{end_index} (xung quanh chunk #{top_index})"

    # Lấy context mở rộng
    context_chunks = []
    (start_index..end_index).each do |i|
      if i == top_index
        # Đánh dấu chunk chính (có similarity cao nhất)
        context_chunks << "**[CHUNK CHÍNH]** #{text_chunks[i]}"
      else
        # Chunk bổ sung cho context
        position = i < top_index ? "TRƯỚC" : "SAU"
        context_chunks << "**[CHUNK #{position}]** #{text_chunks[i]}"
      end
    end

    # Ghép tất cả context chunks thành một đoạn văn liền mạch
    enriched_context = context_chunks.join("\n\n")
    enhanced_chunks << enriched_context

    # Lưu thông tin chi tiết về việc mở rộng
    context_info[top_index] = {
      original_chunk: text_chunks[top_index],
      context_range: [start_index, end_index],
      added_chunks_before: top_index - start_index,
      added_chunks_after: end_index - top_index,
      total_context_chunks: end_index - start_index + 1
    }
  end

  puts "Hoàn thành mở rộng context: #{enhanced_chunks.length} context blocks được tạo"

  {
    enhanced_chunks: enhanced_chunks,
    original_indices: top_indices,
    context_info: context_info
  }
end

# ===============================================================================
# So sánh Context-Enriched vs Basic Search
# ===============================================================================

def compare_search_methods(query, text_chunks, embeddings)
  # So sánh kết quả giữa Basic Search và Context-Enriched Search.
  puts "\n=== So sánh Basic Search vs Context-Enriched Search ==="
  puts "Query: #{query}"
  puts "=" * 80

  # 1. Basic Similarity Search
  puts "\n1. BASIC SIMILARITY SEARCH:"
  puts "-" * 40
  basic_results = basic_similarity_search(query, text_chunks, embeddings, 1)
  if basic_results.any?
    puts "Kết quả Basic Search:"
    puts basic_results[0][0..300] + "..."
    puts "Độ dài: #{basic_results[0].length} ký tự"
  else
    puts "Không tìm thấy kết quả"
  end

  # 2. Context-Enriched Search
  puts "\n2. CONTEXT-ENRICHED SEARCH:"
  puts "-" * 40
  enriched_results = context_enriched_search(query, text_chunks, embeddings, 1, 1)

  if enriched_results[:enhanced_chunks].any?
    puts "Kết quả Context-Enriched Search:"
    puts enriched_results[:enhanced_chunks][0][0..500] + "..."
    puts "Độ dài: #{enriched_results[:enhanced_chunks][0].length} ký tự"

    # Hiển thị thông tin chi tiết về context
    original_index = enriched_results[:original_indices][0]
    context_details = enriched_results[:context_info][original_index]

    puts "\nChi tiết Context Enhancement:"
    puts "- Chunk chính: index #{original_index}"
    puts "- Range mở rộng: #{context_details[:context_range]}"
    puts "- Chunks thêm vào trước: #{context_details[:added_chunks_before]}"
    puts "- Chunks thêm vào sau: #{context_details[:added_chunks_after]}"
    puts "- Tổng chunks trong context: #{context_details[:total_context_chunks]}"
  else
    puts "Không tìm thấy kết quả"
  end

  puts "=" * 80

  { basic: basic_results, enriched: enriched_results }
end

# ===============================================================================
# Tạo phản hồi với Enhanced Context
# ===============================================================================

def generate_enhanced_response(query, enhanced_chunks)
  # Tạo phản hồi AI sử dụng enhanced context.
  system_prompt = <<~PROMPT
  Bạn là một AI assistant thông minh có khả năng đọc hiểu và phân tích văn bản.
  Bạn sẽ nhận được ngữ cảnh đã được mở rộng với các thông tin xung quanh.

  CHÚ Ý:
  - [CHUNK CHÍNH] là thông tin có liên quan trực tiếp nhất với câu hỏi
  - [CHUNK TRƯỚC] và [CHUNK SAU] cung cấp ngữ cảnh bổ sung
  - Hãy sử dụng TẤT CẢ thông tin có sẵn để đưa ra câu trả lời đầy đủ nhất
  - Nếu thông tin không đủ, hãy nói rõ phần nào còn thiếu
  PROMPT

  # Chuẩn bị context từ enhanced chunks
  context = enhanced_chunks.each_with_index.map do |chunk, i|
    "=== CONTEXT BLOCK #{i + 1} ===\n#{chunk}"
  end.join("\n\n")

  user_message = "#{context}\n\n=== CÂU HỎI ===\n#{query}"

  generate_response(system_prompt, user_message)
end

# ===============================================================================
# Demo chính
# ===============================================================================

def run_context_enriched_demo
  # Chạy demo hoàn chỉnh Context-Enriched RAG.
  puts "\n=== Demo Context-Enriched RAG ==="

  # Bước 1: Trích xuất văn bản từ PDF
  puts "\n1. Trích xuất văn bản từ PDF..."
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  puts "Độ dài văn bản: #{extracted_text.length} ký tự"

  # Bước 2: Chia thành chunks
  puts "\n2. Chia văn bản thành chunks..."
  text_chunks = chunk_text(extracted_text, 1000, 200)
  puts "Số chunks được tạo: #{text_chunks.length}"

  # Bước 3: Tạo embeddings
  puts "\n3. Tạo embeddings cho các chunks..."
  chunk_embeddings = create_embeddings(text_chunks)
  puts "Đã tạo #{chunk_embeddings.length} embeddings"

  return unless chunk_embeddings.any?

  # Bước 4: So sánh các phương pháp search
  queries = [
    "What are the applications of AI in healthcare?",
    "How does machine learning differ from traditional programming?",
    "What are the ethical concerns related to artificial intelligence?"
  ]

  queries.each_with_index do |query, i|
    puts "\n" + "=" * 100
    puts "DEMO #{i + 1}: #{query}"
    puts "=" * 100

    # So sánh Basic vs Context-Enriched
    comparison_results = compare_search_methods(query, text_chunks, chunk_embeddings)

    # Tạo phản hồi với Enhanced Context
    if comparison_results[:enriched][:enhanced_chunks].any?
      puts "\n3. PHẢN HỒI VỚI CONTEXT-ENRICHED:"
      puts "-" * 50
      enhanced_response = generate_enhanced_response(query, comparison_results[:enriched][:enhanced_chunks])

      if enhanced_response
        puts enhanced_response
      else
        puts "Không thể tạo phản hồi"
      end
    end

    puts "\n" + "=" * 100
  end
end

# ===============================================================================
# Demonstration các kích thước Context khác nhau
# ===============================================================================

def demonstrate_context_sizes
  # Demo việc thay đổi context_size và ảnh hưởng của nó.
  puts "\n=== Demo các kích thước Context khác nhau ==="

  # Thiết lập dữ liệu
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  text_chunks = chunk_text(extracted_text, 800, 150)  # Chunks nhỏ hơn để demo rõ hơn
  chunk_embeddings = create_embeddings(text_chunks)

  return unless chunk_embeddings.any?

  query = "What are the main benefits of artificial intelligence?"
  context_sizes = [0, 1, 2, 3]  # 0 = basic search, 1-3 = different context sizes

  puts "Query: #{query}"
  puts "Số chunks tổng cộng: #{text_chunks.length}"
  puts "=" * 80

  context_sizes.each do |context_size|
    puts "\n--- CONTEXT SIZE: #{context_size} ---"

    if context_size == 0
      # Basic search (no context)
      results = basic_similarity_search(query, text_chunks, chunk_embeddings, 1)
      puts "Kết quả Basic Search (không có context):"
      puts "Độ dài: #{results[0].length} ký tự" if results.any?
      puts results[0][0..200] + "..." if results.any?
    else
      # Context-enriched search
      results = context_enriched_search(query, text_chunks, chunk_embeddings, 1, context_size)
      if results[:enhanced_chunks].any?
        puts "Kết quả với context_size = #{context_size}:"
        puts "Độ dài: #{results[:enhanced_chunks][0].length} ký tự"

        # Thông tin chi tiết
        original_index = results[:original_indices][0]
        context_info = results[:context_info][original_index]
        puts "Tổng chunks trong context: #{context_info[:total_context_chunks]}"
        puts "Range: #{context_info[:context_range]}"

        puts "\nNội dung (200 ký tự đầu):"
        puts results[:enhanced_chunks][0][0..200] + "..."
      end
    end
    puts "-" * 40
  end
end

# ===============================================================================
# Main Execution
# ===============================================================================

if __FILE__ == $0
  puts "Context-Enriched RAG bằng Ruby"
  puts "=" * 60

  begin
    # Chạy demo chính
    run_context_enriched_demo

    # Demo các kích thước context khác nhau
    puts "\n\n"
    demonstrate_context_sizes

    puts "\n=== HOÀN THÀNH DEMO CONTEXT-ENRICHED RAG ==="
  rescue => e
    puts "Lỗi trong quá trình thực thi: #{e.message}"
    puts e.backtrace
  end
end
