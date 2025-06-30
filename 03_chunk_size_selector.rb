#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================================================
# Đánh giá Kích thước Chunk trong Simple RAG (Phiên bản Ruby)
# ===============================================================================
#
# Việc chọn kích thước chunk phù hợp là rất quan trọng để cải thiện độ chính xác
# truy xuất trong pipeline Retrieval-Augmented Generation (RAG). Mục tiêu là
# cân bằng giữa hiệu suất truy xuất và chất lượng phản hồi.
#
# Script này đánh giá các kích thước chunk khác nhau bằng cách:
#
# 1. Trích xuất văn bản từ PDF.
# 2. Chia văn bản thành các chunk có kích thước khác nhau.
# 3. Tạo embeddings cho mỗi chunk.
# 4. Truy xuất các chunk liên quan cho một query.
# 5. Tạo phản hồi sử dụng các chunk đã truy xuất.
# 6. Đánh giá faithfulness và relevancy.
# 7. So sánh kết quả cho các kích thước chunk khác nhau.

require 'net/http'
require 'uri'
require 'json'
require 'matrix'
require 'pdf-reader'

# ===============================================================================
# Thiết lập Môi trường
# ===============================================================================
puts "=== Đánh giá Kích thước Chunk trong RAG ==="

# Cấu hình
OPENAI_API_KEY = ENV['OPENAI_API_KEY']
BASE_URL = 'https://api.studio.nebius.com/v1/'
EMBEDDING_MODEL = 'BAAI/bge-en-icl'
CHAT_MODEL = 'meta-llama/Llama-3.3-70B-Instruct'

# Đọc file .env nếu tồn tại
if File.exist?('.env')
  File.readlines('.env').each do |line|
    key, value = line.strip.split('=', 2)
    ENV[key] = value if key && value
  end
end

# Hệ thống điểm đánh giá
SCORE_FULL = 1.0     # Hoàn toàn thỏa mãn
SCORE_PARTIAL = 0.5  # Thỏa mãn một phần
SCORE_NONE = 0.0     # Không thỏa mãn

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

def chunk_text(text, chunk_size, overlap)
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
  #
  # Args:
  #   system_prompt (String): Hệ thống prompt.
  #   user_message (String): Tin nhắn người dùng.
  #   model (String): Mô hình AI.
  #
  # Returns:
  #   String: Phản hồi từ mô hình AI.
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
  #
  # Args:
  #   vec1 (Array): Vector thứ nhất.
  #   vec2 (Array): Vector thứ hai.
  #
  # Returns:
  #   Float: Độ tương tự cosine giữa hai vector.
  v1 = Vector[*vec1]
  v2 = Vector[*vec2]

  dot_product = v1.inner_product(v2)
  magnitude1 = Math.sqrt(v1.inner_product(v1))
  magnitude2 = Math.sqrt(v2.inner_product(v2))

  return 0.0 if magnitude1 == 0.0 || magnitude2 == 0.0

  dot_product / (magnitude1 * magnitude2)
end

def retrieve_relevant_chunks(query, text_chunks, chunk_embeddings, k = 5)
  # Truy xuất top-k chunks liên quan nhất.
  #
  # Args:
  #   query (String): Câu hỏi của người dùng.
  #   text_chunks (Array): Mảng các text chunks.
  #   chunk_embeddings (Array): Embeddings của các chunks.
  #   k (Integer): Số lượng chunks cần trả về.
  #
  # Returns:
  #   Array: Các chunks liên quan nhất.
  # Tạo embedding cho query
  query_embeddings = create_embeddings([query])
  return [] if query_embeddings.empty?

  query_embedding = query_embeddings[0]

  # Tính similarity scores
  similarities = chunk_embeddings.each_with_index.map do |chunk_emb, i|
    similarity = cosine_similarity(query_embedding, chunk_emb)
    [i, similarity]
  end

  # Sắp xếp và lấy top k
  top_indices = similarities.sort_by { |_, score| -score }.first(k).map(&:first)
  top_indices.map { |i| text_chunks[i] }
end

def generate_ai_response(query, system_prompt, retrieved_chunks)
  # Tạo phản hồi AI dựa trên retrieved chunks.
  #
  # Args:
  #   query (String): Câu hỏi của người dùng.
  #   system_prompt (String): Hệ thống prompt.
  #   retrieved_chunks (Array): Mảng các chunks liên quan.
  #
  # Returns:
  #   String: Phản hồi AI.
  context = retrieved_chunks.each_with_index.map do |chunk, i|
    "Context #{i + 1}:\n#{chunk}"
  end.join("\n\n")

  user_prompt = "#{context}\n\nQuestion: #{query}"
  generate_response(system_prompt, user_prompt)
end

# ===============================================================================
# Hàm Đánh giá
# ===============================================================================

def evaluate_response(question, response, true_answer)
  # Đánh giá phản hồi AI dựa trên faithfulness và relevancy.
  #
  # Args:
  #   question (String): Câu hỏi của người dùng.
  #   response (String): Phản hồi AI.
  #   true_answer (String): Câu trả lời đúng.
  #
  # Returns:
  #   Hash: {faithfulness: Float, relevancy: Float}
  # Template đánh giá Faithfulness
  faithfulness_prompt = <<~PROMPT
    Đánh giá độ chính xác của phản hồi AI so với câu trả lời đúng.
    Câu hỏi: #{question}
    Phản hồi AI: #{response}
    Câu trả lời đúng: #{true_answer}

    Faithfulness đo lường mức độ phản hồi AI phù hợp với sự thật trong câu trả lời đúng, không có hallucination.

    HƯỚNG DẪN:
    - Chấm điểm CHẶT CHẼ chỉ sử dụng các giá trị sau:
        * #{SCORE_FULL} = Hoàn toàn chính xác, không mâu thuẫn với câu trả lời đúng
        * #{SCORE_PARTIAL} = Chính xác một phần, mâu thuẫn nhỏ
        * #{SCORE_NONE} = Không chính xác, mâu thuẫn lớn hoặc hallucination
    - Chỉ trả về điểm số (#{SCORE_FULL}, #{SCORE_PARTIAL}, hoặc #{SCORE_NONE}) không giải thích thêm.
  PROMPT

  # Template đánh giá Relevancy
  relevancy_prompt = <<~PROMPT
    Đánh giá độ liên quan của phản hồi AI với câu hỏi người dùng.
    Câu hỏi: #{question}
    Phản hồi AI: #{response}

    Relevancy đo lường mức độ phản hồi giải quyết câu hỏi của người dùng.

    HƯỚNG DẪN:
    - Chấm điểm CHẶT CHẼ chỉ sử dụng các giá trị sau:
        * #{SCORE_FULL} = Hoàn toàn liên quan, trả lời trực tiếp câu hỏi
        * #{SCORE_PARTIAL} = Liên quan một phần, giải quyết một số khía cạnh
        * #{SCORE_NONE} = Không liên quan, không giải quyết câu hỏi
    - Chỉ trả về điểm số (#{SCORE_FULL}, #{SCORE_PARTIAL}, hoặc #{SCORE_NONE}) không giải thích thêm.
  PROMPT

  # Đánh giá bằng LLM
  system_prompt = "Bạn là một hệ thống đánh giá khách quan. Chỉ trả về điểm số."

  faithfulness_response = generate_response(system_prompt, faithfulness_prompt)
  relevancy_response = generate_response(system_prompt, relevancy_prompt)

  # Parse điểm số
  begin
    faithfulness_score = faithfulness_response.to_f
  rescue
    puts "Cảnh báo: Không thể parse điểm faithfulness, mặc định là 0"
    faithfulness_score = 0.0
  end

  begin
    relevancy_score = relevancy_response.to_f
  rescue
    puts "Cảnh báo: Không thể parse điểm relevancy, mặc định là 0"
    relevancy_score = 0.0
  end

  {
    faithfulness: faithfulness_score,
    relevancy: relevancy_score
  }
end

# ===============================================================================
# Chạy Demo Chính
# ===============================================================================

def run_chunk_size_evaluation
  # Chạy đánh giá hoàn chỉnh các kích thước chunk khác nhau.
  #
  # Returns:
  #   Hash: Kết quả đánh giá cho các kích thước chunk khác nhau.
  puts "\n=== Demo Chunk Size Selector ==="

  # Bước 1: Trích xuất văn bản từ PDF
  puts "\n1. Trích xuất văn bản từ PDF..."
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  puts "Độ dài văn bản: #{extracted_text.length} ký tự"

  # Bước 2: Tạo chunks với các kích thước khác nhau
  puts "\n2. Tạo chunks với các kích thước khác nhau..."
  chunk_sizes = [128, 256, 512]

  text_chunks_dict = {}
  chunk_sizes.each do |size|
    overlap = size / 5  # 20% overlap
    chunks = chunk_text(extracted_text, size, overlap)
    text_chunks_dict[size] = chunks
    puts "Kích thước chunk: #{size}, Số chunks: #{chunks.length}"
  end

  # Bước 3: Tạo embeddings cho các chunks
  puts "\n3. Tạo embeddings cho các chunks..."
  chunk_embeddings_dict = {}
  chunk_sizes.each do |size|
    puts "Đang tạo embeddings cho chunks kích thước #{size}..."
    embeddings = create_embeddings(text_chunks_dict[size])
    chunk_embeddings_dict[size] = embeddings
    puts "Đã tạo #{embeddings.length} embeddings"
  end

  # Bước 4: Load validation data và thực hiện tìm kiếm
  puts "\n4. Load validation data và thực hiện tìm kiếm..."
  validation_data = load_validation_data('data/val.json')

  if validation_data.empty?
    puts "Không có dữ liệu validation, sử dụng query mẫu"
    query = "How does AI contribute to personalized medicine?"
    true_answer = "AI contributes to personalized medicine by analyzing individual patient data, predicting treatment responses, and tailoring medical interventions to each patient's unique characteristics, genetic profile, and medical history."
  else
    query = validation_data[3]['question']
    true_answer = validation_data[3]['ideal_answer']
  end

  puts "Query: #{query}"

  # Bước 5: Truy xuất chunks và tạo phản hồi cho mỗi kích thước
  puts "\n5. Truy xuất chunks và tạo phản hồi..."
  system_prompt = "Bạn là một AI assistant chỉ trả lời dựa trên ngữ cảnh được cung cấp. Nếu câu trả lời không thể được suy ra trực tiếp từ ngữ cảnh, hãy trả lời: 'Tôi không có đủ thông tin để trả lời câu hỏi đó.'"

  results = {}
  chunk_sizes.each do |size|
    puts "\nĐang xử lý chunk size #{size}..."

    # Truy xuất chunks liên quan
    retrieved_chunks = retrieve_relevant_chunks(
      query,
      text_chunks_dict[size],
      chunk_embeddings_dict[size],
      5
    )

    # Tạo phản hồi AI
    ai_response = generate_ai_response(query, system_prompt, retrieved_chunks)

    if ai_response
      # Đánh giá phản hồi
      evaluation_scores = evaluate_response(query, ai_response, true_answer)

      results[size] = {
        response: ai_response,
        faithfulness: evaluation_scores[:faithfulness],
        relevancy: evaluation_scores[:relevancy],
        retrieved_chunks: retrieved_chunks
      }

      puts "Phản hồi: #{ai_response[0..200]}..."
      puts "Faithfulness: #{evaluation_scores[:faithfulness]}"
      puts "Relevancy: #{evaluation_scores[:relevancy]}"
    else
      puts "Không thể tạo phản hồi cho chunk size #{size}"
    end
  end

  # Bước 6: So sánh kết quả
  puts "\n6. So sánh kết quả các kích thước chunk:"
  puts "=" * 80
  puts sprintf("%-12s %-15s %-15s %-15s", "Chunk Size", "Faithfulness", "Relevancy", "Avg Score")
  puts "=" * 80

  results.each do |size, result|
    avg_score = (result[:faithfulness] + result[:relevancy]) / 2.0
    puts sprintf("%-12d %-15.1f %-15.1f %-15.1f",
                 size,
                 result[:faithfulness],
                 result[:relevancy],
                 avg_score)
  end
  puts "=" * 80

  # Tìm kích thước chunk tốt nhất
  best_size = results.max_by { |_, result| (result[:faithfulness] + result[:relevancy]) / 2.0 }
  if best_size
    puts "\nKích thước chunk tốt nhất: #{best_size[0]} (Điểm trung bình: #{((best_size[1][:faithfulness] + best_size[1][:relevancy]) / 2.0).round(2)})"
  end

  results
end

# ===============================================================================
# Hàm Tiện ích
# ===============================================================================

def load_validation_data(file_path)
  # Tải dữ liệu validation từ file JSON.
  #
  # Args:
  #   file_path (String): Đường dẫn đến file JSON.
  #
  # Returns:
  #   Array: Mảng các dữ liệu validation.
  if File.exist?(file_path)
    JSON.parse(File.read(file_path))
  else
    puts "Không tìm thấy file validation: #{file_path}"
    []
  end
end

# ===============================================================================
# Main Execution
# ===============================================================================

if __FILE__ == $0
  puts "Đánh giá Kích thước Chunk trong RAG bằng Ruby"
  puts "=" * 60

  begin
    results = run_chunk_size_evaluation
    puts "\n=== HOÀN THÀNH ĐÁNH GIÁ CHUNK SIZE ==="
  rescue => e
    puts "Lỗi trong quá trình thực thi: #{e.message}"
    puts e.backtrace
  end
end
