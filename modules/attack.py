import numpy as np
import pandas as pd
from modules.differential_evolution import differential_evolution
from modules.plot import word_to_url_token
import modules.helper as helper
from modules.plot import *


class PixelAttacker:
    def __init__(self,mdl, model_url, model_html, model_dns, c_tk, w_tk,  X_test_dom = None,  X_test_html_w = None,  X_test_html_s = None, X_test_dns_c_ = None, X_test_dns_w_ = None):
        
        self.model = model_url
        self.model_html = model_html
        self.model_dns = model_dns
        self.c_tk = c_tk
        self.w_tk = w_tk
        self.dom = X_test_dom
        self.html_w = X_test_html_w
        self.html_s = X_test_html_s
        self.dns_c = X_test_dns_c_
        self.dns_w = X_test_dns_w_
        self.class_names = ['beging','phishing']
        self.dimension = 200
        self.other_model = other_model
        self.mdl = mdl

    def predict_classes(self, xs, url_id, url, target_class, minimize=True):
        urls_perturbed = helper.perturb_image(xs, url)
        urls = np.reshape(np.array(urls_perturbed),(np.shape(xs)[0], 200))
        c_x = []
        w_x = []
        for i in urls:
            c, w = word_to_url_token(i, self.c_tk, self.w_tk)
            c_x.append(c)
            w_x.append(w)
        tile_shapes = (np.shape(c_x)[0], 1)
        predictions = self.model.predict([c_x, w_x])[:, 0]
        if target_class == 0:
            predictions = 1 - predictions
        return predictions if minimize else 1 - predictions

    def attack_success(self, id, x, url, target_class, targeted_attack=False, verbose=False):
        attack_url = helper.perturb_image(x, url)
        c, w = word_to_url_token(attack_url[0][0][0], self.c_tk, self.w_tk)
        confidence = self.model.predict([[c], [w]])[0]
        predicted_class = 1 if confidence >= 0.5 else 0
        if verbose:
            print('Confidence:', confidence)
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, url_id, x_test, y_test, target=None, pixel_count=1,
               maxiter=75, popsize=400, verbose=False):
        targeted_attack = target is not None
        target_class = target if targeted_attack else y_test
        dim_max = self.dimension
        check = [[8,20,20,16], [8,20,20,16,19]]
        restrict = min([i for i in range(len(x_test)) if x_test[i] != 0])
        dim_min = 7 if set(x_test[restrict:restrict+4])==set(check[0]) else 0
        dim_min = 8 if set(x_test[restrict:restrict+5])==set(check[1]) else dim_min
        dim_min = restrict + dim_min
        bounds = [(dim_min, dim_max), (1, 52)] * pixel_count
        popmul = max(1, popsize // len(bounds))
        def predict_fn(xs):
            return self.predict_classes(xs, url_id, x_test, target_class, minimize=True)

        def callback_fn(x, convergence):
            return self.attack_success(url_id, x, x_test, target_class)
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)
        attack_url = helper.perturb_image(attack_result.x, x_test)[0]
        attack_url = np.reshape(attack_url,(200))
        c, w = word_to_url_token(attack_url, self.c_tk, self.w_tk)
        
        predicted_probs_u = self.model.predict([[c], [w]])[0]
        predicted_class_u = 1 if predicted_probs_u >= 0.5 else 0
        success_u = predicted_class_u != y_test
        
        predicted_probs = self.mdl.predict_pt6(self.model, self.model_dom, self.model_html, self.model_dns,float(0.01), data_url = [[c], [w]], data_html = [[self.html_w[url_id]], [self.html_s[url_id]]], data_dom = [[self.dom[url_id]]], data_dns = [[self.dns_c[url_id]], [self.dns_w[url_id]]])[0]
        predicted_class = 1 if predicted_probs >= 0.5 else 0
        success = predicted_class != y_test
        print("url:", success_u, "seq:", success)
        
        if success:
            plot_ae_url(np.array(c), './result/url2.csv')
        return success_u, success

    def attack_all(self, id_, x_tests, y_tests, pixels=[3], \
                    targeted=False,maxiter=100, popsize=250, verbose=False):
        results = []
        for pixel_count in pixels:
            cnt = 0
            cnt_u = 0
            cnt1 = 0
            for i in id_:
                cnt1 = cnt1 + 1
                print('image-', i + 1, '/', len(id_))
                # c, w = word_to_url_token(x_tests[i], self.c_tk, self.w_tk)
                # confidence = self.model.predict([[c], [w], [self.dom[i]], [self.html_w[i]],[self.html_s[i]]])[0]
                targets = [None] if not targeted else range(2)

                for target in targets:
                    if targeted:
                        print('Attacking with target', self.class_names[target])
                        if target == self.y_test[x_tests[i], 0]:
                            continue
                    result_u, result = self.attack(i, x_tests[i], y_tests[i], target, pixel_count,
                                            maxiter=maxiter, popsize=popsize, verbose=verbose)
                    if result_u:
                        cnt_u = cnt_u + 1
                    if result == True:
                        results.append(i)
                        write_id = str(i) + '\n'
                        plot_ae_result(write_id, './result/id2.csv')
                        cnt = cnt + 1
                print("url:", cnt_u/cnt1, "seq:", cnt/cnt1)
            if pixel_count == 1:
                plot_ae_url([cnt], './result/count.csv')
            else:
                plot_ae_url([cnt], './result/count.csv', sp = ',')
        return cnt/len(id_)