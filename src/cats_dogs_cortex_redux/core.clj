(ns cats-dogs-cortex-redux.core
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [cortex.compute.cpu.driver :as cpu-driver]
            [cortex.compute.cpu.tensor-math :as cpu-tm]
            [cortex.experiment.classification :as classification]
            [cortex.experiment.train :as train]
            [cortex.experiment.util :as experiment-util]
            [cortex.graph :as graph]
            [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.tensor :as ct]
            [cortex.util :as util]
            [mikera.image.core :as i]
            [mikera.image.core :as imagez]
            [think.datatype.core :as dtype]
            [think.image.data-augmentation :as image-aug]
            [think.image.image :as image]
            [think.image.patch :as patch]
            [think.parallel.core :as parallel])
  (:gen-class))


;;; IMAGE CLEANUP

(def original-data-dir "data/train")
(def training-dir "data/cats-dogs-training")
(def testing-dir "data/cats-dogs-testing")
(def test-training-split 0.85)
(def image-size 224)

(defn produce-indexed-data-label-seq
  [files]
  (->> (map (fn [file] [file (-> (.getName file) (string/split #"\.") first)]) files)
       (map-indexed vector)))

(defn resize-and-write-data
  [output-dir [idx [file label]]]
  (let [img-path (str output-dir "/" label "/" idx ".png" )]
    (when-not (.exists (io/file img-path))
      (io/make-parents img-path)
      (-> (imagez/load-image file)
          (image/resize image-size image-size)
          (imagez/save img-path)))
    nil))

(defn- gather-files [path]
  (->> (io/file path)
       (file-seq)
       (filter #(.isFile %))))


(defn build-image-data
  []
  (let [files (gather-files original-data-dir)
        [training-files testing-files] (into [] (partition-all (int (* (count files) test-training-split)) (shuffle files)))
        training-observation-label-seq (produce-indexed-data-label-seq
                                        training-files)
        testing-observation-label-seq (produce-indexed-data-label-seq
                                       testing-files)
        train-fn (partial resize-and-write-data training-dir)
        test-fn (partial resize-and-write-data  testing-dir)]
    (println "Building the image data with a test-training split of " test-training-split)
    (println "training files " (count training-files) "testing files " (count testing-files))
    (dorun (pmap train-fn training-observation-label-seq))
    (dorun (pmap test-fn testing-observation-label-seq))))


;; NETWORK SETUP ;;

(def layers-to-add
  [(layers/linear 2 :id :fc2)
   (layers/softmax :id :labels)])

(defn load-network
  [network-file chop-layer top-layers]
  (let [network (util/read-nippy-file network-file)
        ;; remove last layer(s)
        chopped-net (network/dissoc-layers-from-network network chop-layer)
        ;; set layers to non-trainable
        nodes (get-in chopped-net [:compute-graph :nodes]) ;;=> {:linear-1 {<params>}
        new-node-params (mapv (fn [params] (assoc params :non-trainable? true)) (vals nodes))
        frozen-nodes (zipmap (keys nodes) new-node-params)
        frozen-net (assoc-in chopped-net [:compute-graph :nodes] frozen-nodes)
        ;; add top layers
        modified-net (network/assoc-layers-to-network frozen-net (flatten top-layers))]
    modified-net))


;; DATA SETUP ;;


(defn- load-image
  [path]
  (i/load-image path))

(def train-folder "data/train")
(def test-folder "data/test")


(defn create-train-test-folders
  "Given an original data directory that contains subdirs of classes (e.g. orig/cats, orig/dogs)
  and a split proportion, divide each class of files into a new training and testing directory
  (e.g. train/cats, train/dogs, test/cats, test/dogs)"
  [orig-data-path & {:keys [test-proportion]
                     :or {test-proportion 0.3}}]
  (let [subdirs (->> (file-seq (io/file orig-data-path))
                     (filter #(.isDirectory %) )
                     (map #(.getPath %))
                     ;; remove top (root) directory
                     rest
                     (map (juxt identity gather-files))
                     (filter #(> (count (second %)) 0)))]
    (for [[dir files] subdirs]
      (let [num-test (int (* test-proportion (count files)))
            test-files (take num-test files)
            train-files (drop num-test files)
            copy-fn (fn [file root-path]
                      (let [dest-path (str root-path "/" (last (string/split dir #"/")) "/" (.getName file))]
                        (when-not (.exists (io/file dest-path))
                          (io/make-parents dest-path)
                          (io/copy file (io/file dest-path)))))]
        (println "Working on " dir)
        (dorun (pmap (fn [file] (copy-fn file train-folder)) train-files))
        (dorun (pmap (fn [file] (copy-fn file test-folder)) test-files))
        ))))



;; TRAINING ;;

(defn classes
  []
  (into [] (.list (io/file training-dir))))

(defn class-mapping
  []
  {:class-name->index (zipmap (classes) (range))
   :index->class-name (zipmap (range) (classes))})


(defn check-file-sizes
  []
  (->> (concat (file-seq (io/file training-dir))
               (file-seq (io/file testing-dir)))
       (filter #(.endsWith (.getName %) "png"))
       (remove #(try (let [img (i/load-image %)]
                       (and (= 224 (image/width img))
                            (= 224 (image/height img))))
                     (catch Throwable e
                       (println (format "Failed to load image %s" %))
                       (println e)
                       true)))))


(defn dataset-from-folder
  [folder-name infinite?]
  (cond-> (->> (file-seq (io/file folder-name))
               (filter #(.endsWith ^String (.getName %) "png"))
               (map (fn [file-data]
                      {:class-name (.. file-data getParentFile getName)
                       :file file-data})))
    infinite?
    (experiment-util/infinite-class-balanced-seq :class-key :class-name)))


(defn src-ds-item->net-input
  [{:keys [class-name file] :as entry}]
  (let [img-dim 224
        src-image (i/load-image file)
        ;;Ensure image is correct size
        src-image (if-not (and (= (image/width src-image) img-dim)
                               (= (image/height src-image) img-dim))
                    (i/resize src-image img-dim img-dim)
                    src-image)
        ary-data (image/->array src-image)
        ;;mask out the b-g-r channels
        mask-tensor (-> (ct/->tensor [(bit-shift-left 0xFF 16)
                                      (bit-shift-left 0xFF 8)
                                      0xFF]
                                     :datatype :int)
                        (ct/in-place-reshape [3 1 1]))
        ;;Divide to get back to range of 0-255
        div-tensor (-> (ct/->tensor [(bit-shift-left 1 16)
                                     (bit-shift-left 1 8)
                                     1]
                                    :datatype :int)
                       (ct/in-place-reshape [3 1 1]))
        ;;Use the normalization the network expects
        subtrack-tensor (-> (ct/->tensor [123.68 116.779 103.939])
                            (ct/in-place-reshape [3 1 1]))
        ;;Array of packed integer data
        img-tensor (-> (cpu-tm/as-tensor ary-data)
                       (ct/in-place-reshape [img-dim img-dim]))
        ;;Result will be b-g-r planar data
        intermediate (ct/new-tensor [3 img-dim img-dim] :datatype :int)
        result (ct/new-tensor [3 img-dim img-dim])]
    (ct/binary-op! intermediate 1.0 img-tensor 1.0 mask-tensor :bit-and)
    (ct/binary-op! intermediate 1.0 intermediate 1.0 div-tensor :/)
    (ct/assign! result intermediate)
    ;;Switch to floating point for final subtract
    (ct/binary-op! result 1.0 result 1.0 subtrack-tensor :-)
    (when-not (->> (classes)
                   (filter (partial = class-name))
                   first)
      (throw (ex-info "Class not found in classes"
                      {:classes (classes)
                       :class-name class-name})))
    {:class-name class-name
     :labels (util/one-hot-encode (classes) class-name)
     :data (cpu-tm/as-java-array result)
     :filepath (.getPath file)}))


(defn net-input->image
  [{:keys [data]}]
  (cpu-tm/tensor-context
   (let [img-dim 224
         ;;src is in normalized bgr space
         src-tens (-> (cpu-tm/as-tensor data)
                      (ct/in-place-reshape [3 img-dim img-dim]))
         subtrack-tensor (-> (ct/->tensor [123.68 116.779 103.939] :datatype (dtype/get-datatype src-tens))
                             (ct/in-place-reshape [3 1 1]))
         div-tensor (-> (ct/->tensor [(bit-shift-left 1 16)
                                      (bit-shift-left 1 8)
                                      1]
                                     :datatype (dtype/get-datatype src-tens))
                       (ct/in-place-reshape [3 1 1]))
         intermediate-float (ct/new-tensor [3 img-dim img-dim]
                                           :datatype (dtype/get-datatype src-tens))
         intermediate-int (ct/new-tensor [3 img-dim img-dim]
                                         :datatype :int)
         result (ct/new-tensor [img-dim img-dim] :datatype :int)]
     (ct/binary-op! intermediate-float 1.0 src-tens 1.0 subtrack-tensor :+)
     (ct/binary-op! intermediate-float 1.0 intermediate-float 1.0 div-tensor :*)
     (ct/assign! intermediate-int intermediate-float)
     ;;Sum together to reverse the bit shifting
     (ct/binary-op! result 1.0 result 1.0 intermediate-int :+)
     ;;Add back in alpha else we just get black images
     (ct/binary-op! result 1.0 result 1.0 (bit-shift-left 1 24) :+)
     (image/array-> (image/new-image 224 224) (cpu-tm/as-java-array result)))))


(defn convert-one-ds-item
  [ds-item]
  (src-ds-item->net-input ds-item))


(defn train-ds
  [epoch-size batch-size]
  (when-not (= 0 (rem (long epoch-size)
                      (long batch-size)))
    (throw (ex-info "Batch size is not commensurate with epoch size" {:epoch-size epoch-size
                                                                      :batch-size batch-size})))
  (ct/with-stream (cpu-driver/main-thread-cpu-stream)
   (ct/with-datatype :float
     (->> (dataset-from-folder training-dir true)
          (take epoch-size)
          (parallel/queued-pmap (* 2 batch-size) src-ds-item->net-input)
          vec))))


(defn test-ds
  [batch-size]
  (ct/with-stream (cpu-driver/main-thread-cpu-stream)
   (ct/with-datatype :float
     (->> (dataset-from-folder testing-dir false)
          (experiment-util/batch-pad-seq batch-size)
          (parallel/queued-pmap (* 2 batch-size) src-ds-item->net-input)))))


(defn train
  [& [batch-size]]
  (let [batch-size (or batch-size 32)
        epoch-size 4096
        network (load-network "models/resnet50.nippy" :fc1000 layers-to-add)]
    (println "training using batch size of" batch-size)
    (train/train-n network
                   (partial train-ds epoch-size batch-size)
                   (partial test-ds batch-size)
                   :batch-size batch-size :epoch-count 1)))



(defn get-training-size
  [network batch-size]
  (let [traversal (traverse/training-traversal network)
        buffers (:buffers traversal)
        get-buff-size-fn (fn [buffer] (let [dims (get-in buffer [1 :dimension])]
                                        (* (:channels dims) (:height dims) (:width dims))))
        io-total (reduce + (mapv #(get-buff-size-fn %) buffers))
        param-count (graph/parameter-count (:compute-graph network))
        ;; num-vals: 4 * param-count (params, gradients, two for adam) + 2 * io-total (params, gradients)
        vals-per-batch (+ (* 4 param-count) (* 2 io-total))]
    (println "IO buffers: " io-total)
    (println "Parameter count: " param-count)
    ;; memory: 4 (4 bytes per float) * batch-size * vals-per-batch
    (* 4 batch-size vals-per-batch)))


(defn train-again
  "incrementally improve upon the trained model"
  [& [batch-size]]
  (let [batch-size (or batch-size 32)
        epoch-size 21250
        network (util/read-nippy-file "trained-network.nippy")]
    (println "training using batch size of" batch-size)
    (train/train-n network
                   (partial train-ds epoch-size batch-size)
                   (partial test-ds batch-size)
                   :batch-size batch-size :epoch-count 1)))


(defn get-class [idx]
  "A convienence function to get the class name"
    (get (classes) idx))


(defn label-one
  "Take an arbitrary test image and label it."
  []
  (let [data-item  (rand-nth (test-ds 100))]
    (->> data-item :filepath (i/load-image) (i/show))
    {:answer (->> data-item :labels util/max-index get-class)
     :guess (let [[result] (execute/run (util/read-nippy-file "trained-network.nippy") [data-item])
                  r-idx (util/max-index (:labels result))]
              {:prob (get (:labels result) r-idx) :class (get-class r-idx)})}))

;;; Scoring

(defn score [val]
 ;; clipping for better kaggle result
  (cond
    (>= val 0.95) 0.95
    (<= val 0.05) 0.05
    :else val))

(defn write-kaggle-header []
  (with-open [out-file (io/writer "kaggle-results.csv" :append true)]
    (csv/write-csv out-file
                   [["id" "label"]])))

(defn write-kaggle-results [results]
  (with-open [out-file (io/writer "kaggle-results.csv" :append true)]
    (csv/write-csv out-file
                   (-> (mapv (fn [[id {:keys [prob class]}]]
                               [(Integer/parseInt id) (if (= "dog" class) (score prob) (score (- 1 prob)))]) results)
                       (sort)))))

(defn kaggle-id [filepath]
  (-> filepath (string/split #"\/") last (string/split #"\.") first))

(defn kaggle-test-ds
  [batch-size]
  (ct/with-stream (cpu-driver/main-thread-cpu-stream)
   (ct/with-datatype :float
     (->> (gather-files "data/kaggle-test/test")
          (parallel/queued-pmap (* 2 batch-size) #(src-ds-item->net-input {:class-name "cat" :file %}))
          (partition-all batch-size)))))

(defn classify-kaggle-tests [batch-size]
  (execute/run (util/read-nippy-file "trained-network.nippy")
    (partial kaggle-test-ds batch-size)
    :batch-size batch-size))

(defn kaggle-results [batch-size]
  (spit "kaggle-results.csv" "" :append false)
  (write-kaggle-header)
  (let [data-parts (kaggle-test-ds batch-size)]
    (map (fn [data-part]
           (print ".")
           (let [results (execute/run (util/read-nippy-file "trained-network.nippy") data-part :batch-size batch-size)]
             (->> (map (fn [r data]
                         (let [r-idx  (util/max-index (:labels r))]
                           [(kaggle-id (:filepath data)) {:prob (get (:labels r) r-idx) :class (get-class r-idx)}]))
                       results
                       data-part)
                  (write-kaggle-results))))
         data-parts)))


;;; Main for uberjar

(defn -main
  [& [batch-size continue]]
  (let [batch-size-num (when batch-size (Integer/parseInt batch-size))]
    (if continue
      (do
        (println "Training again....")
        (train-again batch-size-num))
      (do
        (println "Training fresh from RESNET-50")
        (train batch-size-num)))))


(comment


(kaggle-results 5)


  )

