// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include <cmath>


namespace mediapipe {

    namespace {

        const char weights[] =
                "rows: 63\n"
                "cols: 4\n"
                "packed_data: -0.2939860770044298\n"
                "packed_data: 0.09723430308772632\n"
                "packed_data: -5.8114779784705916e-05\n"
                "packed_data: -0.02535958925156674\n"
                "packed_data: -0.026547021982165978\n"
                "packed_data: -0.07325275954928155\n"
                "packed_data: 0.24431599064351167\n"
                "packed_data: -0.1284516440518801\n"
                "packed_data: -0.1088711577216958\n"
                "packed_data: 0.06371041329786808\n"
                "packed_data: -0.2449590849562378\n"
                "packed_data: -0.12390209693974544\n"
                "packed_data: -0.36595148488491874\n"
                "packed_data: -0.1477640727754343\n"
                "packed_data: -0.1188116693589042\n"
                "packed_data: 0.2332856206145177\n"
                "packed_data: 0.2250530337676147\n"
                "packed_data: 0.10994632975312157\n"
                "packed_data: -0.08203527997702165\n"
                "packed_data: -0.04122503170348685\n"
                "packed_data: -0.017141155540407304\n"
                "packed_data: -0.05246368736341544\n"
                "packed_data: 0.18373496726606792\n"
                "packed_data: 0.04619960344185404\n"
                "packed_data: 0.02957296061079147\n"
                "packed_data: 0.08038772189758289\n"
                "packed_data: 0.18368826774031327\n"
                "packed_data: 0.12022514233139368\n"
                "packed_data: 0.15901943265261095\n"
                "packed_data: 0.10974937533788806\n"
                "packed_data: 0.18593050606515515\n"
                "packed_data: 0.02237723655667493\n"
                "packed_data: -0.08454869556532507\n"
                "packed_data: 0.09829728754180764\n"
                "packed_data: 0.24674900111057596\n"
                "packed_data: -0.09694105311734709\n"
                "packed_data: -0.04526658422621021\n"
                "packed_data: -0.06418849321126383\n"
                "packed_data: -0.007597605510823047\n"
                "packed_data: -0.030448663266399233\n"
                "packed_data: -0.006918458660713108\n"
                "packed_data: 0.09018297581083959\n"
                "packed_data: 0.08716237166308607\n"
                "packed_data: -0.18443620200643418\n"
                "packed_data: -0.12109533778586662\n"
                "packed_data: -3.8054374848775756e-05\n"
                "packed_data: 0.0079864246310172\n"
                "packed_data: -0.09946487244570437\n"
                "packed_data: -0.1237527365004774\n"
                "packed_data: -0.1409503906940357\n"
                "packed_data: -0.030641026521276163\n"
                "packed_data: -0.22237915811211725\n"
                "packed_data: -0.2562546153238391\n"
                "packed_data: 0.0718339962970599\n"
                "packed_data: 0.014622690726168813\n"
                "packed_data: -0.16897626528447107\n"
                "packed_data: -0.08591140692208454\n"
                "packed_data: 0.1070244171033012\n"
                "packed_data: 0.0911015907308121\n"
                "packed_data: 0.016689694627373138\n"
                "packed_data: 0.09956910320646109\n"
                "packed_data: 0.14430382771678768\n"
                "packed_data: 0.11651022322663697\n"
                "packed_data: 0.025567947410689174\n"
                "packed_data: 0.07650265388080327\n"
                "packed_data: 0.010443125614813458\n"
                "packed_data: -0.0046401691802440085\n"
                "packed_data: 0.243555464355283\n"
                "packed_data: -0.4092128319185818\n"
                "packed_data: -0.07880010165534487\n"
                "packed_data: -0.7611971021786569\n"
                "packed_data: -0.2339731577765501\n"
                "packed_data: 0.005448210791490093\n"
                "packed_data: 0.5902557676326028\n"
                "packed_data: 0.38797858107436645\n"
                "packed_data: 0.04266977406431992\n"
                "packed_data: -0.1520286126294874\n"
                "packed_data: 0.023640152804657536\n"
                "packed_data: 0.2586780631902106\n"
                "packed_data: -0.031124337295588615\n"
                "packed_data: -0.44611236324844356\n"
                "packed_data: 0.038259779760273414\n"
                "packed_data: 0.30495128780297187\n"
                "packed_data: 0.774625608250122\n"
                "packed_data: -0.14098662014061486\n"
                "packed_data: -0.06103960712101428\n"
                "packed_data: -1.2695309249349347\n"
                "packed_data: 0.16534332356166853\n"
                "packed_data: -0.00792078370000382\n"
                "packed_data: 0.7349992917755245\n"
                "packed_data: -0.6320700796815668\n"
                "packed_data: -0.18236540734410203\n"
                "packed_data: 0.20547805641430836\n"
                "packed_data: 0.22272707479292284\n"
                "packed_data: 0.08777104054288214\n"
                "packed_data: 0.09494771255100744\n"
                "packed_data: -0.13812641541170323\n"
                "packed_data: -0.6776367358853189\n"
                "packed_data: 0.45286612428656037\n"
                "packed_data: -0.0131325656816387\n"
                "packed_data: 0.4704723350914922\n"
                "packed_data: -0.3766632217611846\n"
                "packed_data: 0.4647734027240634\n"
                "packed_data: 0.04979377444639819\n"
                "packed_data: -0.017224809421251045\n"
                "packed_data: -0.5285570663720642\n"
                "packed_data: 0.6308372635668709\n"
                "packed_data: -0.3052053168896781\n"
                "packed_data: 0.5640499501256738\n"
                "packed_data: -0.6125729318634511\n"
                "packed_data: -0.5840162215313851\n"
                "packed_data: -0.3114021161064909\n"
                "packed_data: 0.2441981494133183\n"
                "packed_data: 0.5838723660993764\n"
                "packed_data: 0.10599572108041755\n"
                "packed_data: 0.11258267086570017\n"
                "packed_data: -0.4696122285504879\n"
                "packed_data: -0.18648865793314964\n"
                "packed_data: -1.0315691235465592\n"
                "packed_data: 0.6760269154584998\n"
                "packed_data: 0.15406132176287435\n"
                "packed_data: 1.3369892522573665\n"
                "packed_data: -0.01001936544835204\n"
                "packed_data: -0.013342934779200339\n"
                "packed_data: -0.5923451540771605\n"
                "packed_data: -0.2374873412506417\n"
                "packed_data: -0.20101828245907216\n"
                "packed_data: -0.32179200423676413\n"
                "packed_data: -9.354689429622233e-06\n"
                "packed_data: 0.1341157974021667\n"
                "packed_data: -0.061178292524045906\n"
                "packed_data: 0.18017685619291943\n"
                "packed_data: 0.14689629687643405\n"
                "packed_data: 0.15808934476645556\n"
                "packed_data: 0.08743823525899137\n"
                "packed_data: -0.2897675660664541\n"
                "packed_data: 0.0441362389489192\n"
                "packed_data: 0.002684817069389089\n"
                "packed_data: 0.10439205776693468\n"
                "packed_data: -0.1446887244383349\n"
                "packed_data: 0.09354207496391985\n"
                "packed_data: 0.31451925520679425\n"
                "packed_data: -0.16894783678487355\n"
                "packed_data: 0.010919705071258782\n"
                "packed_data: 0.06761107584291787\n"
                "packed_data: 0.7096339919778875\n"
                "packed_data: 0.12023582352563578\n"
                "packed_data: -0.32741923374285836\n"
                "packed_data: 0.08203058406143642\n"
                "packed_data: -0.3143902188270454\n"
                "packed_data: 0.13955542833524034\n"
                "packed_data: -0.31621214902420647\n"
                "packed_data: -0.764958569474221\n"
                "packed_data: -0.36847892209859945\n"
                "packed_data: -0.13320315618095172\n"
                "packed_data: 0.0908214702802807\n"
                "packed_data: 0.3878722947741917\n"
                "packed_data: -0.2008427560284789\n"
                "packed_data: -0.1400373128876416\n"
                "packed_data: 0.045556037691687534\n"
                "packed_data: -0.37398565623014\n"
                "packed_data: -0.15277413089072298\n"
                "packed_data: -0.07363062405892383\n"
                "packed_data: 0.15291484193614918\n"
                "packed_data: 0.2982464279003806\n"
                "packed_data: -0.5246162641558091\n"
                "packed_data: -0.19919961889033805\n"
                "packed_data: 0.21215522288629057\n"
                "packed_data: 0.25272128265641647\n"
                "packed_data: 0.03567137438008118\n"
                "packed_data: 0.23101234202099913\n"
                "packed_data: 0.15759783040201286\n"
                "packed_data: 0.08096027514765236\n"
                "packed_data: 0.1932295419013112\n"
                "packed_data: 0.2673905719394914\n"
                "packed_data: 0.06399074468432168\n"
                "packed_data: 0.37274805222375984\n"
                "packed_data: 0.17402416496077924\n"
                "packed_data: 0.7624917776596186\n"
                "packed_data: 0.3314461077031336\n"
                "packed_data: 0.06795177633838165\n"
                "packed_data: 0.022089201355727527\n"
                "packed_data: 0.21166896696488835\n"
                "packed_data: -0.1994259614381957\n"
                "packed_data: -0.3004350945418218\n"
                "packed_data: -0.05759887055917021\n"
                "packed_data: -0.2676268583817938\n"
                "packed_data: 0.03516047795004511\n"
                "packed_data: -0.14579096690282103\n"
                "packed_data: -0.21899350456464461\n"
                "packed_data: -0.08611754753916484\n"
                "packed_data: -0.006705608281127654\n"
                "packed_data: 0.23328395113552836\n"
                "packed_data: -0.44040385843625307\n"
                "packed_data: 0.8095800574543358\n"
                "packed_data: -0.3753426191735986\n"
                "packed_data: 0.8243468433192198\n"
                "packed_data: 0.2779740423314168\n"
                "packed_data: 0.5829836593216431\n"
                "packed_data: -0.31643146740990064\n"
                "packed_data: -0.2905210915442936\n"
                "packed_data: -0.375512823078596\n"
                "packed_data: 0.01514049631819873\n"
                "packed_data: -0.30582570555737043\n"
                "packed_data: 0.4885056046749353\n"
                "packed_data: -0.08526833408874175\n"
                "packed_data: -0.10313048854782537\n"
                "packed_data: -0.43394831416880053\n"
                "packed_data: -0.4524693275580188\n"
                "packed_data: 0.7590408138743738\n"
                "packed_data: 0.7932545320107586\n"
                "packed_data: 0.13591209155787523\n"
                "packed_data: 0.8246846634210173\n"
                "packed_data: -0.5391882351256211\n"
                "packed_data: -0.048379893079876585\n"
                "packed_data: -0.9312840763016041\n"
                "packed_data: -0.6079021395305816\n"
                "packed_data: 0.3064325087396284\n"
                "packed_data: -0.5605413902260922\n"
                "packed_data: -0.3910547974977102\n"
                "packed_data: 0.3750788437503751\n"
                "packed_data: -0.40703192300261504\n"
                "packed_data: 0.13971485944716894\n"
                "packed_data: 0.5092901355686379\n"
                "packed_data: 0.035480145205683065\n"
                "packed_data: 0.11640265692439898\n"
                "packed_data: -0.43884529020455476\n"
                "packed_data: 0.07127291505069287\n"
                "packed_data: 0.23601129515015987\n"
                "packed_data: -0.27679487623802224\n"
                "packed_data: -0.2561042740162541\n"
                "packed_data: 0.5329622307384485\n"
                "packed_data: -0.608628880419949\n"
                "packed_data: -0.10888326744148616\n"
                "packed_data: 0.0641924987726172\n"
                "packed_data: -0.2982122845271812\n"
                "packed_data: 0.044208082963237985\n"
                "packed_data: -0.15869607723669615\n"
                "packed_data: 0.22879497663709247\n"
                "packed_data: 0.08480065630036229\n"
                "packed_data: 0.05129352235358517\n"
                "packed_data: 0.15867280214740295\n"
                "packed_data: 1.1497898508372546\n"
                "packed_data: 0.33458455083371513\n"
                "packed_data: 0.475208353297948\n"
                "packed_data: -0.32902545115798604\n"
                "packed_data: -1.12992740562698\n"
                "packed_data: -0.1593579747184018\n"
                "packed_data: -0.16139239279657017\n"
                "packed_data: 0.6505053540069456\n"
                "packed_data: 0.1466336784142029\n"
                "packed_data: 0.2532719994273686\n"
                ;

        const char test[] =
                "rows: 2\n"
                "cols: 1\n"
                "packed_data: 0.1\n"
                "packed_data: 0.2\n";

        const char _A[] =
                "rows: 3\n"
                "cols: 2\n"
                "packed_data: 1\n"
                "packed_data: 2\n"
                "packed_data: 3\n"
                "packed_data: -1\n"
                "packed_data: 0\n"
                "packed_data: 0\n"
                ;

        const char _B[] =
                "rows: 2\n"
                "cols: 2\n"
                "packed_data: 1\n"
                "packed_data: 2\n"
                "packed_data: 1\n"
                "packed_data: 0\n";


        constexpr char kLandmarksTag[] = "LANDMARKS";
        constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
        constexpr char textTag[] = "TEXT";


        class PredictSymbolCalculator : public CalculatorBase {
        public:
            PredictSymbolCalculator() {}

            ~PredictSymbolCalculator() override {}

            PredictSymbolCalculator(const PredictSymbolCalculator &) =
            delete;

            PredictSymbolCalculator &operator=(
                    const PredictSymbolCalculator &) = delete;

            static ::mediapipe::Status GetContract(CalculatorContract *cc);

            ::mediapipe::Status Open(CalculatorContext *cc) override;

            ::mediapipe::Status Process(CalculatorContext *cc) override;

        };

        REGISTER_CALCULATOR(PredictSymbolCalculator);


        void MatrixFromTextProto(const std::string& text_proto, Matrix* matrix) {
            CHECK(matrix);
            MatrixData matrix_data;
            CHECK(proto_ns::TextFormat::ParseFromString(text_proto, &matrix_data));
            MatrixFromMatrixDataProto(matrix_data, matrix);
        }




        ::mediapipe::Status PredictSymbolCalculator::GetContract(
                CalculatorContract *cc) {

            RET_CHECK(cc->Inputs().HasTag(kLandmarksTag) ||
                      cc->Inputs().HasTag(kNormLandmarksTag))
                        << "None of the input streams are provided.";
            RET_CHECK(!(cc->Inputs().HasTag(kLandmarksTag) &&
                        cc->Inputs().HasTag(kNormLandmarksTag)))
                        << "Can only one type of landmark can be taken. Either absolute or "
                           "normalized landmarks.";

            if (cc->Inputs().HasTag(kLandmarksTag)) {
                cc->Inputs().Tag(kLandmarksTag).Set<std::vector<Landmark>>();
            }
            if (cc->Inputs().HasTag(kNormLandmarksTag)) {
                cc->Inputs().Tag(kNormLandmarksTag).Set<std::vector<NormalizedLandmark>>();
            }
            cc->Outputs().Tag(textTag).Set<std::string>();
            return ::mediapipe::OkStatus();
        }








        ::mediapipe::Status PredictSymbolCalculator::Open(
                CalculatorContext *cc) {
            cc->SetOffset(TimestampDiff(0));

            return ::mediapipe::OkStatus();
        }






        ::mediapipe::Status PredictSymbolCalculator::Process(
                CalculatorContext *cc) {

            std::string out_text = "";
            double counter = 0.0;

            Matrix* weight_matrix = new Matrix();
            MatrixFromTextProto(weights, weight_matrix);




            auto matrix = new Matrix(1, 63);

            if (cc->Inputs().HasTag(kLandmarksTag)) {
                const auto &landmarks =
                        cc->Inputs().Tag(kLandmarksTag).Get<std::vector<Landmark>>();

                int i = 0;
                for (const auto &landmark : landmarks) {
//                    out_text += std::to_string(landmark.x()) + " " + std::to_string(landmark.y())  + " " +std::to_string(landmark.z()) + "\n";
//                    counter += landmark.x() + landmark.y();
                    (*matrix)(0, i*3 + 0) = landmark.x();
                    (*matrix)(0, i*3 + 1) = landmark.y();
                    (*matrix)(0, i*3 + 2) = landmark.z();
                    i++;
                }

            }

            if (cc->Inputs().HasTag(kNormLandmarksTag)) {
                const auto &landmarks = cc->Inputs()
                        .Tag(kNormLandmarksTag)
                        .Get<std::vector<NormalizedLandmark>>();

                int i = 0;
                for (const auto &landmark : landmarks) {
//                    out_text += std::to_string(landmark.x()) + " " + std::to_string(landmark.y())  + " " +std::to_string(landmark.z()) + "\n";
//                    counter += landmark.x() + landmark.y();
                    (*matrix)(0, i*3 + 0) = landmark.x();
                    (*matrix)(0, i*3 + 1) = landmark.y();
                    (*matrix)(0, i*3 + 2) = landmark.z();
                    i++;
                }

            }


            Matrix* multiplied = new Matrix();
            *multiplied =  (*matrix)  * (*weight_matrix);


            (*multiplied)(0, 0) = (*multiplied)(0, 0) - 0.004022874980557995979;
            (*multiplied)(0, 1) = (*multiplied)(0, 1) + 2.698146777990787726;
            (*multiplied)(0, 2) = (*multiplied)(0, 2) - 0.1184259554171348261;
            (*multiplied)(0, 3) = (*multiplied)(0, 3) - 1.090657369844077573;



            int max = -1000000;
            int num = -1;
            for (int i = 0; i < 4; i++){
                if ((*multiplied)(0, i) > max){
                    max = (*multiplied)(0, i);
                    num = i;
                }
            }

            if (max < 0)
                num = -1;

            if ((*multiplied)(0, 0) < -100)
                num = 1;

            if ((*matrix)(0, 0) == 0.0)
                num = -1;

            switch (num) {
                case -1:
                    out_text = "  ";
                    break;
                case 0:
                    out_text = "A";
                    break;
                case 1:
                    out_text = " ";
                    break;
                case 2:
                    out_text = "H";
                    break;
                case 3:
                    out_text = "Я";
                    break;
            }



//            out_text = std::to_string((*multiplied)(0, 0)) + " " + std::to_string((*multiplied)(0, 1))  + " " +std::to_string((*multiplied)(0, 2)) + " " + std::to_string((*multiplied)(0, 3)) + "\n";
//            out_text = std::to_string(counter);
//            out_text = std::to_string((*matrix)(0, 0));



            auto res_data = absl::make_unique<std::string>(out_text);


            cc->Outputs()
                    .Tag(textTag)
                    .Add(res_data.release(), cc->InputTimestamp());
            return ::mediapipe::OkStatus();
        }


    }
}// namespace mediapipe
