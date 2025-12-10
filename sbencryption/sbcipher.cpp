#include <iostream>
#include <iomanip>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include <algorithm>
#include <random>
#include <limits>
#include <cmath>
using namespace std;

// S-box substitution table for Advance Encryption Stadard
const unsigned char SB_AES[] = {
   0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
   0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
   0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
   0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
   0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
   0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
   0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
   0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
   0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
   0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
   0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
   0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
   0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
   0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
   0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
   0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

const unsigned char SB_PWL[] = {
   0xf3, 0x82, 0xee, 0xaa, 0x9d, 0x55, 0x97, 0x8b, 0xa4, 0xfd, 0x59, 0xc9, 0x46, 0x2a, 0xe6, 0x1c, 0x4a, 0x00, 0x1f, 0x78, 0x65, 0x4f, 0x8d, 0x6a, 0x7d, 0x77, 0xf1, 0x01, 0x79, 0x5b, 0xa1, 0x64, 0xc7, 0x33, 0x96, 0xbf, 0x48, 0xcf, 0xd6, 0x15, 0xda, 0x31, 0xb7, 0x50, 0xb3, 0xa5, 0x09, 0xc3, 0xd1, 0x22, 0x8a, 0xc0, 0x63, 0x06, 0x73, 0x41, 0x98, 0xae, 0x07, 0x7c, 0x14, 0xc5, 0x9b, 0x11, 0x6d, 0x43, 0xd0, 0xa3, 0x02, 0x45, 0x91, 0xfc, 0x2c, 0x4b, 0x0c, 0x3c, 0xed, 0xb9, 0x16, 0x24, 0xd8, 0x25, 0x68, 0xa8, 0x3b, 0xbb, 0x80, 0xa6, 0xe9, 0xd7, 0x76, 0x6f, 0xce, 0x54, 0xd5, 0x86, 0x95, 0xab, 0x49, 0xf8, 0x20, 0x32, 0x21, 0x94, 0x9f, 0x83, 0x8f, 0x67, 0x2b, 0x47, 0x3d, 0xc6, 0x85, 0x58, 0x28, 0x38, 0x72, 0xf5, 0x88, 0xc8, 0xbe, 0xca, 0xb5, 0x6b, 0x1d, 0x5e, 0x57, 0x6e, 0x0b, 0x1a, 0xf4, 0xff, 0xf6, 0xf7, 0x0e, 0x6c, 0xba, 0x3e, 0xeb, 0xde, 0xb6, 0xe8, 0x8c, 0x3f, 0x62, 0xc1, 0x03, 0x7e, 0xdd, 0xf0, 0x61, 0x18, 0xd4, 0x71, 0x42, 0x3a, 0xea, 0xfe, 0x9e, 0x53, 0x66, 0x13, 0x87, 0x5f, 0x9c, 0x17, 0xfa, 0x7b, 0x29, 0xa2, 0xdb, 0xb2, 0x04, 0x2f, 0x7f, 0x4c, 0x2d, 0x37, 0xcb, 0xcc, 0xec, 0x36, 0xfb, 0x52, 0xb0, 0xdf, 0x8e, 0x1e, 0xb1, 0x90, 0xb8, 0xd9, 0xa0, 0x4e, 0x99, 0xe7, 0x81, 0x26, 0x92, 0x84, 0xc2, 0x74, 0x40, 0x93, 0x10, 0x0d, 0x5c, 0xbc, 0x08, 0x60, 0xe3, 0x12, 0xdc, 0xbd, 0xac, 0x34, 0xad, 0xa7, 0xe5, 0xd2, 0x23, 0x9a, 0x05, 0x44, 0x7a, 0xaf, 0x35, 0xf9, 0xa9, 0x30, 0xc4, 0xcd, 0x39, 0x75, 0x70, 0x0a, 0xb4, 0x19, 0xe2, 0xd3, 0xe0, 0x51, 0x27, 0xf2, 0x5d, 0x1b, 0x69, 0x5a, 0x4d, 0x56, 0xe1, 0x89, 0xe4, 0x0f, 0xef, 0x2e
};

double pwlm(double x, double m1 = 0.8, double m2 = 5, double b1 = 40.8) {
    double a = b1 / m1;
    double b2 = b1 * (m2 / m1);
    
    if (x <= -a) return m1 * x + b1;
    else if (x > -a && x < 0) return m2 * x + b2;
    else if (x >= 0 && x < a) return m2 * x - b2;
    else return m1 * x - b1;
}

class Sbox {
private:
    vector<unsigned char> table;
    vector<unsigned char> original_table;
    int rot_cont;

public:
    Sbox(const vector<unsigned char>& t) : table(t), original_table(t), rot_cont(0) {}
    
    void rotation(int dir = 1, int k = 1) {
        if (dir) { // Right rotation
            rotate(table.rbegin(), table.rbegin() + k, table.rend());
            rot_cont++;
        } else { // Left rotation  
            std::rotate(table.begin(), table.begin() + k, table.end());
            rot_cont--;
        }
    }
    
    void reset_table() {
        table = original_table;
        rot_cont = 0;
    }
    
    // Getters
    const vector<unsigned char>& getTable() const { return table; }
    int getRotCount() const { return rot_cont; }
};

vector<unsigned char> generate_sbox(double x01, double x02, int delay = 10, 
                                        vector<double> f1 = {0.8, 5, 40.8}, 
                                        vector<double> f2 = {0.9, 4, 31}) {
    auto startT = chrono::high_resolution_clock::now();
    
    vector<double> x1 = {x01};
    vector<double> x2 = {x02};
    int m1_seq = 0;
    int m2_seq = 0;
    int delayhf = delay / 2;
    vector<unsigned char> sb;
    
    int i = 0;
    while (sb.size() < 256) {
        x1.push_back(pwlm(x1[i], f1[0], f1[1], f1[2]));
        x2.push_back(pwlm(x2[i], f2[0], f2[1], f2[2]));
        
        if (i >= delay) {
            m1_seq = static_cast<int>(x1[i - delay] + x1[i - delayhf] + x1[i]) % 256;
            m2_seq = static_cast<int>(x2[i - delay] + x2[i - delayhf + 1] + x2[i]) % 256;
            int Zi = static_cast<int>(floor((m1_seq + m2_seq) % 256));
            
            if (find(sb.begin(), sb.end(), Zi) == sb.end()) {
                sb.push_back(static_cast<unsigned char>(Zi));
            }
        }
        i++;
    }
    
    auto endT = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endT - startT);
    cout << "Time: " << duration.count() / 1000000.0 << " seconds" << endl;
    
    return sb;
}

void subst_image(const cv::Mat& input,cv::Mat& output, const unsigned char SB[256]) {
   output.create(input.rows, input.cols, CV_8UC1);
   for (int i = 0; i < input.rows; i++) {
      for (int j = 0; j < input.cols; j++) {
         output.at<unsigned char>(i, j) = SB[input.at<unsigned char>(i, j)];
      }
   }
   
}

cv::Mat enigmarot_cipher(const cv::Mat& image, 
                        const vector<pair<double, double>>& xinits,
                        const vector<vector<double>>& params) {
    auto startT = chrono::high_resolution_clock::now();
    int k = 3;
    
    // Generate S-boxes
    vector<Sbox> sboxes;
    for (int i = 0; i < k; i++) {
        vector<unsigned char> sbox_table = generate_sbox(xinits[i].first, xinits[i].second, 10, params[0], params[1]);
        sboxes.push_back(Sbox(sbox_table));
    }
    
    cv::Mat cipher_arr = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    for (int c = 0; c < image.rows; c++) {
        for (int r = 0; r < image.cols; r++) {
            unsigned char pixel_val = image.at<unsigned char>(c, r);
            unsigned char aux = sboxes[0].getTable()[pixel_val];
            aux = sboxes[1].getTable()[aux];
            aux = sboxes[2].getTable()[aux];
            cipher_arr.at<unsigned char>(c, r) = aux;
            
            // Rotate first S-box
            sboxes[0].rotation();
            
            // Cascade rotation conditions
            if (sboxes[0].getRotCount() % 256 == 0 && sboxes[0].getRotCount() != 0) {
                sboxes[1].rotation();
            }
            
            if (sboxes[1].getRotCount() % 256 == 0 && sboxes[1].getRotCount() != 0) {
                sboxes[2].rotation();
                sboxes[1].reset_table(); // Reset middle S-box rotation counter
            }
            
            if (sboxes[0].getRotCount() % 256 == 0 && sboxes[0].getRotCount() != 0) {
                cout << "Full rotation cycle completed for all S-boxes: " << sboxes[2].getRotCount() << std::endl;
            }
        }
    }
    
    auto endT = chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<chrono::microseconds>(endT - startT);
    cout << "Time: " << duration.count() / 1000000.0 << " seconds" << endl;
    
    cout << "End, rotcont: " << sboxes[0].getRotCount() << " " 
              << sboxes[1].getRotCount() << " " << sboxes[2].getRotCount() << endl;
    
    // Reset S-boxes to original state
    for (auto& sbox : sboxes) {
        sbox.reset_table();
    }
    
    return cipher_arr;
}



vector<int> kmedoids_clustering(const vector<unsigned char>& data, int k = 2, int max_iters = 20) {
    int n = data.size();
    if (n == 0) return vector<int>();
    
    vector<int> labels(n, 0);
    vector<int> medoids(k);
    
    // Initialize medoids using k-means++ approach for better starting points
    random_device rd;
    mt19937 gen(rd());
    
    // First medoid random
    uniform_int_distribution<> dis(0, n - 1);
    medoids[0] = dis(gen);
    
    // Subsequent medoids using k-means++ initialization
    for (int i = 1; i < k; i++) {
        vector<double> distances(n, numeric_limits<double>::max());
        
        for (int j = 0; j < n; j++) {
            for (int m = 0; m < i; m++) {
                double dist = abs(data[j] - data[medoids[m]]);
                if (dist < distances[j]) {
                    distances[j] = dist;
                }
            }
        }
        
        // Convert to probabilities
        double sum_sq_dist = 0.0;
        for (double dist : distances) {
            sum_sq_dist += dist * dist;
        }
        
        if (sum_sq_dist > 0) {
            uniform_real_distribution<> prob_dis(0.0, sum_sq_dist);
            double threshold = prob_dis(gen);
            
            double cumulative = 0.0;
            for (int j = 0; j < n; j++) {
                cumulative += distances[j] * distances[j];
                if (cumulative >= threshold) {
                    medoids[i] = j;
                    break;
                }
            }
        } else {
            medoids[i] = dis(gen);
        }
    }
    
    bool changed = true;
    int iter = 0;
    
    while (changed && iter < max_iters) {
        changed = false;
        
        // Assignment step - assign each point to closest medoid
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            int best_cluster = 0;
            int best_distance = numeric_limits<int>::max();
            
            for (int j = 0; j < k; j++) {
                int dist = abs(static_cast<int>(data[i]) - static_cast<int>(data[medoids[j]]));
                if (dist < best_distance) {
                    best_distance = dist;
                    best_cluster = j;
                }
            }
            
            if (labels[i] != best_cluster) {
                labels[i] = best_cluster;
                changed = true;
            }
        }
        
        // Update step - find new medoids for each cluster
        for (int cluster = 0; cluster < k; cluster++) {
            // Find all points in this cluster
            vector<int> cluster_points;
            for (int i = 0; i < n; i++) {
                if (labels[i] == cluster) {
                    cluster_points.push_back(i);
                }
            }
            
            if (cluster_points.empty()) {
                // If cluster is empty, choose random point
                medoids[cluster] = dis(gen);
                continue;
            }
            
            // Find the point that minimizes total distance to all other points in cluster
            int best_medoid = medoids[cluster];
            int best_total_distance = numeric_limits<int>::max();
            
            // Limit the search to a subset for performance
            int search_size = min(static_cast<int>(cluster_points.size()), 1000);
            for (int idx = 0; idx < search_size; idx++) {
                int candidate = cluster_points[idx];
                int total_distance = 0;
                
                // Calculate total distance to other points in cluster
                for (int other_idx = 0; other_idx < min(static_cast<int>(cluster_points.size()), 500); other_idx++) {
                    total_distance += abs(static_cast<int>(data[candidate]) - static_cast<int>(data[cluster_points[other_idx]]));
                }
                
                if (total_distance < best_total_distance) {
                    best_total_distance = total_distance;
                    best_medoid = candidate;
                }
            }
            
            if (best_medoid != medoids[cluster]) {
                medoids[cluster] = best_medoid;
                changed = true;
            }
        }
        
        iter++;
    }
    
    return labels;
}
vector<double> generate_chaotic_sequence(double x01, double x02, int length, 
                                        vector<double> f1 = {0.8, 5, 40.8}, 
                                        vector<double> f2 = {0.9, 4, 31}) {
    vector<double> x1 = {x01};
    vector<double> x2 = {x02};
    vector<double> sequence;
    
    // Pre-generate enough chaotic values
    int pregen_length = length + 1000; // Generate extra to ensure good distribution
    
    for (int i = 0; i < pregen_length && sequence.size() < length; i++) {
        double next_x1 = pwlm(x1.back(), f1[0], f1[1], f1[2]);
        double next_x2 = pwlm(x2.back(), f2[0], f2[1], f2[2]);
        
        x1.push_back(next_x1);
        x2.push_back(next_x2);
        
        // Combine and normalize to [0,1]
        double combined_val = fmod(fabs(next_x1 + next_x2), 1.0);
        sequence.push_back(combined_val);
    }
    
    // If we still don't have enough, fill with remaining values
    while (sequence.size() < length) {
        double val = fmod(static_cast<double>(sequence.size()) / length, 1.0);
        sequence.push_back(val);
    }
    
    // Trim to exact length
    if (sequence.size() > length) {
        sequence.resize(length);
    }
    
    return sequence;
}

cv::Mat scramble_image_kmedoids(const cv::Mat& image, double x01, double x02, 
                               vector<double> f1 = {0.8, 5, 40.8}, 
                               vector<double> f2 = {0.9, 4, 31}) {
    int rows = image.rows;
    int cols = image.cols;
    int total_pixels = rows * cols;
    
    cout << "Starting image scrambling..." << endl;
    cout << "Image size: " << rows << "x" << cols << " (" << total_pixels << " pixels)" << endl;
    
    // For large images, use a more efficient approach
    // if (total_pixels > 10000) {
    //     cout << "Large image detected, using optimized scrambling..." << endl;
    //     return fast_scramble_image(image, x01, x02, f1, f2);
    // }
    
    // Flatten the image with progress indication
    vector<unsigned char> flat_image;
    flat_image.reserve(total_pixels);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat_image.push_back(image.at<unsigned char>(i, j));
        }
    }
    
    cout << "Applying k-medoids clustering..." << endl;
    
    // Apply k-medoids clustering (k=2 as in the article)
    vector<int> labels = kmedoids_clustering(flat_image, 2);
    
    cout << "Clustering completed, creating clusters..." << endl;
    
    // Create clusters
    vector<unsigned char> cluster0, cluster1;
    for (int i = 0; i < total_pixels; i++) {
        if (labels[i] == 0) {
            cluster0.push_back(flat_image[i]);
        } else {
            cluster1.push_back(flat_image[i]);
        }
    }
    
    // Create P3 by combining clusters
    vector<unsigned char> P3;
    P3.reserve(total_pixels);
    P3.insert(P3.end(), cluster0.begin(), cluster0.end());
    P3.insert(P3.end(), cluster1.begin(), cluster1.end());
    
    cout << "Generating chaotic sequence..." << endl;
    
    // Generate chaotic sequence for scrambling
    vector<double> chaotic_seq = generate_chaotic_sequence(x01, x02, total_pixels, f1, f2);
    
    cout << "Sorting indices..." << endl;
    
    // Get sorting indices from chaotic sequence
    vector<size_t> indices(total_pixels);
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), 
         [&chaotic_seq](size_t i, size_t j) { 
             return chaotic_seq[i] < chaotic_seq[j]; 
         });
    
    cout << "Applying scrambling..." << endl;
    
    // Scramble P3 using the chaotic indices
    vector<unsigned char> P5(total_pixels);
    for (size_t i = 0; i < indices.size(); i++) {
        P5[i] = P3[indices[i]];
    }
    
    // Reshape back to image
    cv::Mat scrambled_image(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scrambled_image.at<unsigned char>(i, j) = P5[i * cols + j];
        }
    }
    
    cout << "Scrambling completed!" << endl;
    
    return scrambled_image;
}

// Fast version for large images

bool save_image(const cv::Mat& img, const string& path) {
    try {
        cv::imwrite(path, img);
        return true;
    } catch (const cv::Exception& ex) {
        cerr << "Exception converting image to PNG format: " << ex.what() << endl;
        return false;
    }

}


int main() {
   cv::Mat image = cv::imread("/Users/gaelsegura/Documents/codes/proyectos/research_repos/test_images/baboon512x512.jpg", cv::IMREAD_GRAYSCALE);

   if (image.empty()) {
      cerr << "Error: Could not open or find the image!" << endl;
      return -1;
   }

   std::vector<std::vector<double>> params = {
    {0.8, 5, 40.8},  // f1 parameters
    {0.9, 4, 31}     // f2 parameters
    };

   // Define initial conditions and parameters
   vector<std::pair<double, double>> xinits = {
      {0.1, 0.2}, 
      {0.3, 0.4}, 
      {0.5, 0.6}
   };

    double x01 = 0.1, x02 = 0.2;
    vector<double> f1_params = {0.8, 5, 40.8};
    vector<double> f2_params = {0.9, 4, 31};

    // Apply k-medoids scrambling
    cv::Mat scrambled = scramble_image_kmedoids(image, x01, x02, f1_params, f2_params);

    cv::imshow("Original", image);
    save_image(image, "original.png");



    cv::imshow("Scrambled (K-medoids)", scrambled);
    save_image(scrambled, "scrambled.png");
    // cv::waitKey(0);

    // return 0;


   // Apply Enigma rotation cipher
   cv::Mat encrypted_image = enigmarot_cipher(scrambled, xinits, params);
   cv::imshow("Encrypted Image", encrypted_image);
   save_image(encrypted_image, "ciphered.png");
   cv::waitKey(0);
   return 0;


   // // Generate custom S-box using PWLM
   // vector<double> f1_params = {0.8, 5, 40.8};
   // vector<double> f2_params = {0.9, 4, 31};
   // vector<unsigned char> custom_sbox = generate_sbox(0.1, 0.2, 10, f1_params, f2_params);

   // // Convert vector to array for use with subst_image
   // unsigned char S_CUSTOM[256];
   // copy(custom_sbox.begin(), custom_sbox.end(), S_CUSTOM);

   // // Apply custom S-box to image
   // cv::Mat customSubstituted;
   // subst_image(image, customSubstituted, S_CUSTOM);
   // cv::imshow("Custom PWLM S-box", customSubstituted);

   // cv::Mat substitutedImage;
   // auto start = chrono::high_resolution_clock::now();
   // subst_image(image, SB_AES,substitutedImage);
   // auto end = chrono::high_resolution_clock::now();
   // double time_taken = chrono::duration_cast<chrono::nanoseconds>(end-start).count();
   // time_taken *= 1e-9;
   // cout << "Elapsed time: " << time_taken; 
   // // Mostrar imágenes
   // cv::imshow("Original Image", image);
   // cv::imshow("Substituted Image", substitutedImage);

}
