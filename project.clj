(defproject cats-dogs-cortex-redux "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-alpha17"]
                 [thinktopic/experiment "0.9.22"]
                 [org.clojure/data.csv "0.1.3"]]
  :uberjar-name "cats-dogs-cortex-redux.jar"
  :jvm-opts ["-Xmx3000m"]
  :plugins [[lein-jupyter "0.1.8"]]
  :main cats-dogs-cortex-redux.core)
